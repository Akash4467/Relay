package sim

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Server is the stand-in's OpenAI-compatible API: POST /v1/chat/completions (SSE when stream=true),
// GET /v1/models, GET /metrics, POST /admin/config.
type Server struct {
	cfg      *Config
	limits   map[string]*Limits
	fixtures map[string]Fixture
	planner  *Planner
	inflight atomic.Int64
	served   atomic.Int64
	rejected atomic.Int64
	logMu    sync.Mutex
	log      *os.File
	Now      func() time.Time
	Sleep    func(r *http.Request, d time.Duration) bool // false if the client went away
}

func NewServer(cfg *Config, fixtures map[string]Fixture, logPath string) (*Server, error) {
	s := &Server{cfg: cfg, fixtures: fixtures, planner: NewPlanner(cfg), limits: map[string]*Limits{},
		Now: time.Now, Sleep: sleepCtx}
	now := s.Now()
	for id, m := range cfg.Models {
		s.limits[id] = NewLimits(m.RPM, m.RPD, m.TPM, m.TPD, now)
	}
	if logPath != "" {
		if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
			return nil, err
		}
		f, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644) // append only: raw results
		if err != nil {
			return nil, err
		}
		s.log = f
	}
	return s, nil
}

func sleepCtx(r *http.Request, d time.Duration) bool {
	if d <= 0 {
		return true
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-r.Context().Done():
		return false
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.chat)
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("GET /metrics", s.metrics)
	mux.HandleFunc("POST /admin/config", s.admin)
	return mux
}

type chatRequest struct {
	Model    string           `json:"model"`
	Messages []map[string]any `json:"messages"`
	Stream   bool             `json:"stream"`
}

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	received := s.Now()
	rec := map[string]any{"received_at": received.UTC().Format(time.RFC3339Nano)}
	defer s.writeLog(rec)
	s.inflight.Add(1)
	defer s.inflight.Add(-1)

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.fail(w, rec, http.StatusBadRequest, "invalid_request_error", "invalid JSON body")
		return
	}
	rec["model"] = req.Model
	limits, ok := s.limits[req.Model]
	if !ok {
		s.fail(w, rec, http.StatusNotFound, "invalid_request_error", fmt.Sprintf("The model `%s` does not exist", req.Model))
		return
	}
	qid := r.Header.Get("X-Relay-Question-Id")
	rec["question_id"] = qid
	fx, ok := s.fixtures[qid]
	if !ok { // Mode A replies only from recorded answers: an unknown question is a bug in the caller, not data
		s.fail(w, rec, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("no fixture for question id %q", qid))
		return
	}
	redraft := IsRedraft(req.Messages)
	draft := fx.Pick(redraft)
	prompt := s.cfg.PromptTokens(req.Messages)
	rec["redraft"], rec["prompt_tokens"], rec["completion_tokens"] = redraft, prompt, draft.CompletionTokens

	// The provider cannot know the completion length up front; the stand-in charges the recorded one at admission.
	if rej := limits.Admit(prompt+draft.CompletionTokens, s.Now()); rej != nil {
		s.rejected.Add(1)
		s.rateLimited(w, rec, req.Model, limits, rej)
		return
	}
	plan := s.planner.Plan(int(s.inflight.Load())-1, draft.ReasoningTokens, draft.CompletionTokens-draft.ReasoningTokens)
	rec["fault"], rec["planned_first_byte_ms"], rec["planned_reasoning_ms"] = plan.Fault, ms2(plan.FirstByte), ms2(plan.Reasoning)
	if plan.Fault == "500" {
		s.fail(w, rec, http.StatusInternalServerError, "internal_server_error", "simulated upstream failure")
		return
	}
	if plan.Fault == "timeout" { // never answer; the client's timeout or disconnect ends it
		rec["status"] = 0
		s.Sleep(r, 10*time.Minute)
		return
	}
	for k, v := range limits.Headers(s.Now()) {
		w.Header().Set(k, v)
	}
	if !req.Stream {
		if !s.Sleep(r, plan.FirstByte+plan.Reasoning) {
			rec["status"] = 499
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "standin", "object": "chat.completion", "model": req.Model,
			"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": draft.Content}, "finish_reason": "stop"}},
			"usage":   usage(prompt, draft)})
		rec["status"] = 200
		return
	}
	s.stream(w, r, rec, req.Model, draft, prompt, plan, received)
}

func (s *Server) stream(w http.ResponseWriter, r *http.Request, rec map[string]any, model string, draft Draft,
	prompt float64, plan Plan, received time.Time) {
	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	if !s.Sleep(r, plan.FirstByte) {
		rec["status"] = 499
		return
	}
	w.WriteHeader(http.StatusOK)
	send := func(v any) bool {
		b, _ := json.Marshal(v)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return false
		}
		if flusher != nil {
			flusher.Flush()
		}
		return true
	}
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{"id": "standin", "object": "chat.completion.chunk", "model": model,
			"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}}}
	}
	send(chunk(map[string]any{"role": "assistant", "content": ""}, nil))
	rec["first_byte_ms"] = ms2(s.Now().Sub(received))
	if !s.Sleep(r, plan.Reasoning) { // hidden reasoning: tokens are spent, nothing is shown
		rec["status"] = 499
		return
	}
	pieces := split(draft.Content, s.cfg.Latency.ChunkTokens, s.cfg.CharsPerToken)
	for i, piece := range pieces {
		if i == 0 {
			rec["first_content_ms"] = ms2(s.Now().Sub(received))
		} else if !s.Sleep(r, plan.ChunkGap) {
			rec["status"] = 499
			return
		}
		if plan.Fault == "stall" && i == plan.StallAfter && !s.Sleep(r, plan.Stall) {
			rec["status"] = 499
			return
		}
		if !send(chunk(map[string]any{"content": piece}, nil)) {
			rec["status"] = 499
			return
		}
	}
	send(chunk(map[string]any{}, "stop"))
	send(map[string]any{"id": "standin", "object": "chat.completion.chunk", "model": model,
		"choices": []any{}, "usage": usage(prompt, draft)}) // top-level usage, as Groq sent it
	fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
	s.served.Add(1)
	rec["status"], rec["total_ms"] = 200, ms2(s.Now().Sub(received))
}

func usage(prompt float64, d Draft) map[string]any {
	return map[string]any{"prompt_tokens": prompt, "completion_tokens": d.CompletionTokens,
		"total_tokens":              prompt + d.CompletionTokens,
		"completion_tokens_details": map[string]any{"reasoning_tokens": d.ReasoningTokens}}
}

// split cuts content into pieces of about chunkTokens tokens, never inside a UTF-8 character.
func split(content string, chunkTokens int, charsPerToken float64) []string {
	runes := []rune(content)
	size := int(math.Max(1, float64(chunkTokens)*charsPerToken))
	var out []string
	for i := 0; i < len(runes); i += size {
		out = append(out, string(runes[i:min(i+size, len(runes))]))
	}
	if len(out) == 0 {
		out = []string{""}
	}
	return out
}

// rateLimited answers like Groq's documented 429: retry-after in seconds and the x-ratelimit-* headers.
// The JSON body copies Groq's public error format; no real 429 body has been observed yet.
func (s *Server) rateLimited(w http.ResponseWriter, rec map[string]any, model string, l *Limits, rej *Rejection) {
	for k, v := range l.Headers(s.Now()) {
		w.Header().Set(k, v)
	}
	retry := int(math.Ceil(rej.RetryAfter.Seconds()))
	w.Header().Set("retry-after", fmt.Sprint(retry))
	msg := fmt.Sprintf("Rate limit reached for model `%s` in organization `org_standin` service tier `on_demand` on %s: "+
		"Limit %.0f, Used %.0f, Requested %.0f. Please try again in %s.", model, rej.Limit, rej.Cap, rej.Used,
		rej.Requested, GroqDuration(rej.RetryAfter))
	rec["limit"], rec["retry_after_s"], rec["per_day"] = rej.Limit, retry, rej.PerDay
	s.fail(w, rec, http.StatusTooManyRequests, rej.Code, msg, "rate_limit_exceeded")
}

func (s *Server) fail(w http.ResponseWriter, rec map[string]any, status int, typ, msg string, code ...string) {
	body := map[string]any{"message": msg, "type": typ}
	if len(code) > 0 {
		body["code"] = code[0]
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": body})
	rec["status"], rec["error"] = status, msg
}

func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	ids := make([]string, 0, len(s.limits))
	for id := range s.limits {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	data := []map[string]any{}
	for _, id := range ids {
		data = append(data, map[string]any{"id": id, "object": "model"})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
}

func (s *Server) metrics(w http.ResponseWriter, _ *http.Request) {
	now := s.Now()
	levels := map[string]any{}
	for id, l := range s.limits {
		l.mu.Lock()
		levels[id] = map[string]float64{"rpm": l.RPM.Available(now), "rpd": l.RPD.Available(now),
			"tpm": l.TPM.Available(now), "tpd": l.TPD.Available(now), "tpm_cap": l.TPM.Cap, "rpm_cap": l.RPM.Cap}
		l.mu.Unlock()
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"inflight": s.inflight.Load(), "served": s.served.Load(),
		"rejected": s.rejected.Load(), "buckets": levels})
}

type adminRequest struct {
	RPMScale     float64 `json:"rpm_scale"`
	TPMScale     float64 `json:"tpm_scale"`
	ExhaustDaily bool    `json:"exhaust_daily"`
	Reset        bool    `json:"reset"`
}

// admin changes the simulated limits for a scenario, e.g. {"rpm_scale":0.5,"tpm_scale":0.5} or {"exhaust_daily":true}.
func (s *Server) admin(w http.ResponseWriter, r *http.Request) {
	var a adminRequest
	if err := json.NewDecoder(r.Body).Decode(&a); err != nil || a.RPMScale < 0 || a.TPMScale < 0 {
		http.Error(w, "bad admin request", http.StatusBadRequest)
		return
	}
	for _, l := range s.limits {
		l.Configure(a.RPMScale, a.TPMScale, a.ExhaustDaily, a.Reset, s.Now())
	}
	s.writeLog(map[string]any{"admin": a, "at": s.Now().UTC().Format(time.RFC3339Nano)})
	s.metrics(w, r)
}

func (s *Server) writeLog(rec map[string]any) {
	if s.log == nil {
		return
	}
	b, _ := json.Marshal(rec)
	s.logMu.Lock()
	defer s.logMu.Unlock()
	_, _ = s.log.Write(append(b, '\n'))
}

func ms2(d time.Duration) float64 { return math.Round(float64(d.Microseconds())/100) / 10 }
