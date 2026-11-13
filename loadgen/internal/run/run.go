// Package run is the open-loop load generator (DESIGN §17).
//
// Arrival times are drawn up front from piecewise Poisson rates with a fixed seed and requests are sent at those
// times whatever happens to earlier ones (no coordinated omission). Latencies are measured from the scheduled time.
// Each simulated user gives up at its own patience (log-normal) and is recorded as having given up.
package run

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type Phase struct {
	FromS      float64 `json:"from_s"`
	ToS        float64 `json:"to_s"`
	RatePerMin float64 `json:"rate_per_min"`
}

type Patience struct {
	MedianMs float64 `json:"median"`
	Sigma    float64 `json:"sigma"`
	MinMs    float64 `json:"min"`
	MaxMs    float64 `json:"max"`
}

type AdminStep struct {
	AtS  float64         `json:"at_s"`
	Body json.RawMessage `json:"body"`
}

type Scenario struct {
	Name             string      `json:"name"`
	Source           string      `json:"source"`
	Target           string      `json:"target"`
	StandIn          string      `json:"standin"`
	QuestionsFile    string      `json:"questions_file"`
	QuestionIDs      []string    `json:"question_ids"`
	RetrievalSetting string      `json:"retrieval_setting"`
	DurationS        float64     `json:"duration_s"`
	Phases           []Phase     `json:"phases"`
	Patience         Patience    `json:"patience_ms"`
	Seed             uint64      `json:"seed"`
	Admin            []AdminStep `json:"admin"`
}

func LoadScenario(path string) (*Scenario, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var s Scenario
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if s.Name == "" || s.Target == "" || len(s.QuestionIDs) == 0 || len(s.Phases) == 0 {
		return nil, fmt.Errorf("%s: name, target, question_ids and phases are required", path)
	}
	return &s, nil
}

// Arrival is one scheduled request.
type Arrival struct {
	Index      int
	At         time.Duration
	QuestionID string
	Patience   time.Duration
}

// Schedule draws every arrival up front: exponential gaps at each phase's rate, a question and a patience each.
func (s *Scenario) Schedule() []Arrival {
	rng := rand.New(rand.NewPCG(s.Seed, s.Seed+1))
	var out []Arrival
	for _, p := range s.Phases {
		if p.RatePerMin <= 0 {
			continue
		}
		t := p.FromS
		for {
			t += rng.ExpFloat64() / (p.RatePerMin / 60)
			if t >= p.ToS || t >= s.DurationS {
				break
			}
			pat := s.Patience.MedianMs * math.Exp(s.Patience.Sigma*rng.NormFloat64())
			pat = math.Min(math.Max(pat, s.Patience.MinMs), s.Patience.MaxMs)
			out = append(out, Arrival{At: time.Duration(t * float64(time.Second)),
				QuestionID: s.QuestionIDs[rng.IntN(len(s.QuestionIDs))], Patience: time.Duration(pat * float64(time.Millisecond))})
		}
	}
	for i := range out {
		out[i].Index = i
	}
	return out
}

// Record is one request as the user saw it.
type Record struct {
	RequestID      string   `json:"request_id"`
	Scenario       string   `json:"scenario"`
	Source         string   `json:"source"`
	QuestionID     string   `json:"question_id"`
	ScheduledMs    float64  `json:"scheduled_offset_ms"`
	StartTime      string   `json:"start_time"`
	PatienceMs     float64  `json:"patience_ms"`
	SendDelayMs    float64  `json:"send_delay_ms"`
	Admitted       *bool    `json:"admitted"`
	FirstContentMs *float64 `json:"first_content_ms"`
	TotalMs        float64  `json:"total_ms"`
	GaveUp         bool     `json:"gave_up"`
	HTTPStatus     int      `json:"http_status"`
	DoneStatus     string   `json:"done_status"`
	DoneReason     string   `json:"done_reason"`
	DoneTier       string   `json:"done_tier"`
	Error          string   `json:"error,omitempty"`
}

// Summary counts outcomes as the brief asks: offered, admitted, completed, rejected, timed out / given up.
type Summary struct {
	Scenario      string  `json:"scenario"`
	Offered       int     `json:"offered"`
	Admitted      int     `json:"admitted"`
	Completed     int     `json:"completed"`
	Rejected      int     `json:"rejected"`
	TimedOut      int     `json:"timed_out"`
	GaveUp        int     `json:"gave_up"`
	Errors        int     `json:"errors"`
	DurationS     float64 `json:"duration_s"`
	OfferedPerMin float64 `json:"offered_per_min"`
}

func Summarize(name string, recs []Record, duration float64) Summary {
	s := Summary{Scenario: name, Offered: len(recs), DurationS: duration}
	if duration > 0 {
		s.OfferedPerMin = math.Round(float64(len(recs))/duration*60*100) / 100
	}
	for _, r := range recs {
		if r.Admitted != nil && *r.Admitted {
			s.Admitted++
		}
		switch {
		case r.GaveUp:
			s.GaveUp++
		case r.Error != "" || r.DoneStatus == "error":
			s.Errors++
		case r.DoneStatus == "ok":
			s.Completed++
		case r.DoneStatus == "rate_limited":
			s.Rejected++
		case r.DoneStatus == "timeout":
			s.TimedOut++
		default:
			s.Errors++
		}
	}
	return s
}

type askBody struct {
	Question         string `json:"question"`
	QuestionID       string `json:"question_id"`
	RequestID        string `json:"request_id"`
	DeadlineMs       int64  `json:"deadline_ms"`
	RetrievalSetting string `json:"retrieval_setting,omitempty"`
	Scenario         string `json:"scenario"`
}

// One sends a request and reads the SSE reply until done or until the user's patience runs out.
func One(client *http.Client, s *Scenario, a Arrival, question string, scheduled time.Time) (rec Record) {
	rec = Record{RequestID: fmt.Sprintf("%s-%04d", s.Name, a.Index), Scenario: s.Name, Source: s.Source,
		QuestionID: a.QuestionID, ScheduledMs: ms(a.At), StartTime: scheduled.UTC().Format(time.RFC3339Nano),
		PatienceMs: ms(a.Patience), SendDelayMs: ms(time.Since(scheduled))}
	ctx, cancel := context.WithDeadline(context.Background(), scheduled.Add(a.Patience))
	defer cancel()
	body, _ := json.Marshal(askBody{Question: question, QuestionID: a.QuestionID, RequestID: rec.RequestID,
		DeadlineMs: a.Patience.Milliseconds(), RetrievalSetting: s.RetrievalSetting, Scenario: s.Name})
	req, _ := http.NewRequestWithContext(ctx, "POST", s.Target, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	defer func() { rec.TotalMs = ms(time.Since(scheduled)) }()
	resp, err := client.Do(req)
	if err != nil {
		rec.GaveUp, rec.Error = errors.Is(ctx.Err(), context.DeadlineExceeded), err.Error()
		if rec.GaveUp {
			rec.Error = ""
		}
		return rec
	}
	defer resp.Body.Close()
	rec.HTTPStatus = resp.StatusCode
	if resp.StatusCode != 200 {
		rec.Error = fmt.Sprintf("HTTP %d", resp.StatusCode)
		return rec
	}
	event := ""
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			event = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			var d map[string]any
			_ = json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &d)
			switch event {
			case "meta":
				adm, _ := d["admitted"].(bool)
				rec.Admitted = &adm
			case "content":
				if rec.FirstContentMs == nil {
					v := ms(time.Since(scheduled))
					rec.FirstContentMs = &v
				}
			case "done":
				rec.DoneStatus, _ = d["status"].(string)
				rec.DoneReason, _ = d["reason"].(string)
				rec.DoneTier, _ = d["tier"].(string)
				return rec
			}
		}
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		rec.GaveUp = true // the user left before the reply finished
	} else if err := sc.Err(); err != nil {
		rec.Error = err.Error()
	} else {
		rec.Error = "stream ended without done"
	}
	return rec
}

func ms(d time.Duration) float64 { return math.Round(float64(d.Microseconds())/100) / 10 }

// LoadQuestions maps question_id -> question text from eval/questions.jsonl.
func LoadQuestions(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[string]string{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var q struct{ QuestionID, Question string }
		var raw map[string]any
		if err := json.Unmarshal(sc.Bytes(), &raw); err != nil {
			return nil, err
		}
		q.QuestionID, _ = raw["question_id"].(string)
		q.Question, _ = raw["question"].(string)
		out[q.QuestionID] = q.Question
	}
	return out, sc.Err()
}

// Run executes the scenario and writes {outDir}/{name}_{ts}.jsonl and .summary.json.
func Run(s *Scenario, questions map[string]string, outDir string, logf func(string, ...any)) (Summary, string, error) {
	for _, id := range s.QuestionIDs {
		if _, ok := questions[id]; !ok {
			return Summary{}, "", fmt.Errorf("question %s not in %s", id, s.QuestionsFile)
		}
	}
	arrivals := s.Schedule()
	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 256}}
	start := time.Now().Add(500 * time.Millisecond)
	var wg sync.WaitGroup
	for _, step := range s.Admin {
		wg.Add(1)
		go func(step AdminStep) {
			defer wg.Done()
			time.Sleep(time.Until(start.Add(time.Duration(step.AtS * float64(time.Second)))))
			resp, err := client.Post(s.StandIn+"/admin/config", "application/json", bytes.NewReader(step.Body))
			if err != nil {
				logf("admin step at %.0fs failed: %v", step.AtS, err)
				return
			}
			resp.Body.Close()
			logf("admin step at %.0fs: %s -> HTTP %d", step.AtS, step.Body, resp.StatusCode)
		}(step)
	}
	recs := make([]Record, len(arrivals))
	for i, a := range arrivals {
		wg.Add(1)
		go func(i int, a Arrival) {
			defer wg.Done()
			scheduled := start.Add(a.At)
			time.Sleep(time.Until(scheduled))
			recs[i] = One(client, s, a, questions[a.QuestionID], scheduled)
		}(i, a)
	}
	wg.Wait()
	sum := Summarize(s.Name, recs, s.DurationS)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return sum, "", err
	}
	base := filepath.Join(outDir, fmt.Sprintf("%s_%s", s.Name, start.UTC().Format("2006-01-02T15-04-05")))
	var buf bytes.Buffer
	for _, r := range recs {
		b, _ := json.Marshal(r)
		buf.Write(append(b, '\n'))
	}
	if err := os.WriteFile(base+".jsonl", buf.Bytes(), 0o644); err != nil {
		return sum, "", err
	}
	b, _ := json.MarshalIndent(sum, "", "  ")
	return sum, base + ".jsonl", os.WriteFile(base+".summary.json", b, 0o644)
}
