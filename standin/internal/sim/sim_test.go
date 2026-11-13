package sim

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

var t0 = time.Date(2026, 11, 13, 12, 0, 0, 0, time.UTC)

func TestBucketRefillsContinuouslyAndCaps(t *testing.T) {
	b := NewBucket(60, time.Minute, t0)
	b.Take(60, t0)
	if got := b.Available(t0.Add(30 * time.Second)); got < 29.99 || got > 30.01 {
		t.Fatalf("after 30 s want 30, got %v", got)
	}
	if got := b.Available(t0.Add(time.Hour)); got != 60 {
		t.Fatalf("capped at 60, got %v", got)
	}
	if w := b.WaitFor(10, t0.Add(time.Hour)); w != 0 {
		t.Fatalf("no wait when available, got %v", w)
	}
}

func TestDailyRequestBucketRefillsOneRequestPer86Point4Seconds(t *testing.T) {
	b := NewBucket(1000, 24*time.Hour, t0)
	b.Empty(t0)
	if w := b.WaitFor(1, t0); w < 86*time.Second || w > 87*time.Second {
		t.Fatalf("observed on Groq: one request back every 86.4 s, got %v", w)
	}
}

func TestAdmitIsAllOrNothingAndNamesTheBindingLimit(t *testing.T) {
	l := NewLimits(2, 1000, 1000, 200000, t0)
	if l.Admit(400, t0) != nil || l.Admit(400, t0) != nil {
		t.Fatal("first two requests fit")
	}
	rej := l.Admit(10, t0)
	if rej == nil || rej.Limit != "requests per minute (RPM)" || rej.PerDay {
		t.Fatalf("third request must hit RPM, got %+v", rej)
	}
	if got := l.TPM.Available(t0); got < 199.99 || got > 200.01 {
		t.Fatalf("a rejected request takes nothing, tpm %v", got)
	}
	rej = NewLimits(30, 1000, 1000, 200000, t0).Admit(5000, t0)
	if rej == nil || rej.Limit != "tokens per minute (TPM)" || rej.RetryAfter != time.Minute {
		t.Fatalf("a request larger than the bucket can never fit, got %+v", rej)
	}
}

func TestConfigureHalvesPerMinuteAndExhaustsDaily(t *testing.T) {
	l := NewLimits(30, 1000, 8000, 200000, t0)
	l.Configure(0.5, 0.5, false, false, t0)
	if l.RPM.Cap != 15 || l.TPM.Cap != 4000 || l.TPM.Available(t0) != 4000 {
		t.Fatalf("halved: %v %v %v", l.RPM.Cap, l.TPM.Cap, l.TPM.Available(t0))
	}
	l.Configure(0, 0, true, false, t0)
	rej := l.Admit(10, t0)
	if rej == nil || !rej.PerDay || rej.Limit != "requests per day (RPD)" {
		t.Fatalf("exhausted daily budget must refuse per day, got %+v", rej)
	}
	l.Configure(0, 0, false, true, t0)
	if l.Admit(10, t0) != nil || l.TPM.Cap != 8000 {
		t.Fatal("reset restores the base limits")
	}
}

func TestGroqDurationFormat(t *testing.T) {
	for d, want := range map[time.Duration]string{
		705 * time.Millisecond: "705ms", 7660 * time.Millisecond: "7.66s", 86400 * time.Millisecond: "1m26.4s",
		17 * time.Second: "17s"} {
		if got := GroqDuration(d); got != want {
			t.Errorf("%v: want %s got %s", d, want, got)
		}
	}
}

func testConfig() *Config {
	c := &Config{Seed: 7, RequestOverheadTokens: 67, CharsPerToken: 4, FixturesDir: "", LogDir: ""}
	c.Latency.FirstByteLogMu, c.Latency.FirstByteLogSigma = 6.243, 0.368
	c.Latency.ReasoningTPS, c.Latency.ContentTPS, c.Latency.ChunkTokens = 280, 530, 8
	c.Models = map[string]struct {
		RPM float64 `json:"rpm"`
		RPD float64 `json:"rpd"`
		TPM float64 `json:"tpm"`
		TPD float64 `json:"tpd"`
	}{"m": {RPM: 30, RPD: 1000, TPM: 8000, TPD: 200000}}
	return c
}

func TestPlannerIsDeterministicForASeedAndSlowsWithLoad(t *testing.T) {
	a, b := NewPlanner(testConfig()), NewPlanner(testConfig())
	for i := 0; i < 20; i++ {
		if pa, pb := a.Plan(0, 70, 230), b.Plan(0, 70, 230); pa != pb {
			t.Fatalf("same seed must give the same plan: %+v vs %+v", pa, pb)
		}
	}
	cfg := testConfig()
	cfg.Latency.LoadRate = 0.1
	p := NewPlanner(cfg)
	idle, busy := p.Plan(0, 280, 0).Reasoning, p.Plan(10, 280, 0).Reasoning
	if idle != time.Second || busy != 2*time.Second {
		t.Fatalf("280 reasoning tokens at 280/s = 1 s, doubled at 10 in flight with c=0.1: %v %v", idle, busy)
	}
}

func TestSplitNeverBreaksCharactersAndKeepsText(t *testing.T) {
	text := "Mimi → 12.5 Hz, “quoted” text"
	parts := split(text, 2, 4)
	if strings.Join(parts, "") != text {
		t.Fatal("pieces must rejoin to the original")
	}
	for _, p := range parts {
		if !strings.Contains(text, p) {
			t.Fatalf("broken piece %q", p)
		}
	}
}

func TestFixturePicksRecordedRedraft(t *testing.T) {
	f := Fixture{Drafts: []Draft{{Content: "first"}, {Content: "second"}}}
	if f.Pick(false).Content != "first" || f.Pick(true).Content != "second" {
		t.Fatal("redraft must replay the recorded redraft")
	}
	if (Fixture{Drafts: []Draft{{Content: "only"}}}).Pick(true).Content != "only" {
		t.Fatal("no recorded redraft: replay the first answer")
	}
	msgs := []map[string]any{{"role": "user", "content": "q"}, {"role": "user", "content": "Your previous answer had unsupported sentences:\n- x"}}
	if !IsRedraft(msgs) || IsRedraft(msgs[:1]) {
		t.Fatal("redraft detection")
	}
}

func newTestServer(t *testing.T, cfg *Config) (*Server, *httptest.Server) {
	fx := map[string]Fixture{"q05": {QuestionID: "q05", Drafts: []Draft{
		{Content: "The routing module uses cosine similarity [HNET:2.2.1:c023].", CompletionTokens: 40, ReasoningTokens: 10}}}}
	s, err := NewServer(cfg, fx, "")
	if err != nil {
		t.Fatal(err)
	}
	s.Sleep = func(*http.Request, time.Duration) bool { return true } // no real waiting in tests
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(ts.Close)
	return s, ts
}

func post(t *testing.T, url, qid, body string) *http.Response {
	req, _ := http.NewRequest("POST", url+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("X-Relay-Question-Id", qid)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

const chatBody = `{"model":"m","stream":true,"messages":[{"role":"user","content":"How does the routing module work?"}]}`

func TestStreamFramingUsageAndHeaders(t *testing.T) {
	_, ts := newTestServer(t, testConfig())
	resp := post(t, ts.URL, "q05", chatBody)
	if resp.StatusCode != 200 || resp.Header.Get("x-ratelimit-limit-tokens") != "8000" || resp.Header.Get("x-ratelimit-limit-requests") != "1000" {
		t.Fatalf("status %d headers %v", resp.StatusCode, resp.Header)
	}
	var text strings.Builder
	var usage map[string]any
	done := false
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			done = true
			break
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("bad SSE data %q", data)
		}
		if u, ok := chunk["usage"].(map[string]any); ok {
			usage = u
		}
		for _, c := range chunk["choices"].([]any) {
			if s, ok := c.(map[string]any)["delta"].(map[string]any)["content"].(string); ok {
				text.WriteString(s)
			}
		}
	}
	if !done || text.String() != "The routing module uses cosine similarity [HNET:2.2.1:c023]." {
		t.Fatalf("stream text %q done=%v", text.String(), done)
	}
	// prompt = ceil(33 chars / 4) + 67 overhead; completion from the recorded fixture
	if usage["prompt_tokens"].(float64) != 76 || usage["completion_tokens"].(float64) != 40 || usage["total_tokens"].(float64) != 116 {
		t.Fatalf("usage %v", usage)
	}
}

func TestUnknownQuestionAndModelAreClientErrors(t *testing.T) {
	_, ts := newTestServer(t, testConfig())
	if r := post(t, ts.URL, "q99", chatBody); r.StatusCode != 400 {
		t.Fatalf("unknown fixture must be 400, got %d", r.StatusCode)
	}
	if r := post(t, ts.URL, "q05", strings.Replace(chatBody, `"m"`, `"nope"`, 1)); r.StatusCode != 404 {
		t.Fatalf("unknown model must be 404, got %d", r.StatusCode)
	}
}

func TestRateLimited429LooksLikeGroq(t *testing.T) {
	cfg := testConfig()
	m := cfg.Models["m"]
	m.RPM = 1
	cfg.Models["m"] = m
	_, ts := newTestServer(t, cfg)
	if r := post(t, ts.URL, "q05", chatBody); r.StatusCode != 200 {
		t.Fatal("first request fits")
	}
	r := post(t, ts.URL, "q05", chatBody)
	if r.StatusCode != 429 || r.Header.Get("retry-after") == "" || r.Header.Get("x-ratelimit-remaining-tokens") == "" {
		t.Fatalf("want 429 with retry-after and rate headers, got %d %v", r.StatusCode, r.Header)
	}
	var body struct {
		Error struct{ Message, Type, Code string } `json:"error"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	if body.Error.Code != "rate_limit_exceeded" || body.Error.Type != "requests" ||
		!strings.Contains(body.Error.Message, "requests per minute (RPM)") {
		t.Fatalf("body %+v", body.Error)
	}
}

func TestAdminExhaustDailyGivesPerDay429(t *testing.T) {
	_, ts := newTestServer(t, testConfig())
	resp, err := http.Post(ts.URL+"/admin/config", "application/json", strings.NewReader(`{"exhaust_daily":true}`))
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("admin: %v %v", err, resp)
	}
	r := post(t, ts.URL, "q05", chatBody)
	var body struct {
		Error struct{ Message string } `json:"error"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	if r.StatusCode != 429 || !strings.Contains(body.Error.Message, "per day") {
		t.Fatalf("want per-day 429, got %d %q", r.StatusCode, body.Error.Message)
	}
}

func TestInjected500(t *testing.T) {
	cfg := testConfig()
	cfg.Faults.P500 = 1
	_, ts := newTestServer(t, cfg)
	if r := post(t, ts.URL, "q05", chatBody); r.StatusCode != 500 {
		t.Fatalf("want 500, got %d", r.StatusCode)
	}
}
