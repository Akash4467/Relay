package run

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func scenario() *Scenario {
	return &Scenario{Name: "t", Source: "stand-in", QuestionIDs: []string{"q05", "q11"}, DurationS: 600, Seed: 42,
		Phases:   []Phase{{FromS: 0, ToS: 60, RatePerMin: 6}, {FromS: 60, ToS: 90, RatePerMin: 120}, {FromS: 90, ToS: 600, RatePerMin: 6}},
		Patience: Patience{MedianMs: 15000, Sigma: 0.5, MinMs: 3000, MaxMs: 60000}}
}

func TestScheduleIsDeterministicOrderedPerPhaseAndRespectsPatienceBounds(t *testing.T) {
	a, b := scenario().Schedule(), scenario().Schedule()
	if len(a) != len(b) || len(a) == 0 {
		t.Fatalf("same seed, same schedule: %d vs %d", len(a), len(b))
	}
	burst := 0
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("arrival %d differs", i)
		}
		if a[i].Patience < 3*time.Second || a[i].Patience > 60*time.Second {
			t.Fatalf("patience out of bounds: %v", a[i].Patience)
		}
		if a[i].At >= 60*time.Second && a[i].At < 90*time.Second {
			burst++
		}
	}
	// 120/min for 30 s: about 60 arrivals expected; a Poisson count far outside 40..80 means a broken rate
	if burst < 40 || burst > 80 {
		t.Fatalf("burst phase produced %d arrivals", burst)
	}
}

func TestScheduleRateMatchesOnAverage(t *testing.T) {
	total := 0
	for seed := uint64(1); seed <= 40; seed++ {
		s := &Scenario{Name: "r", QuestionIDs: []string{"q"}, DurationS: 600, Seed: seed,
			Phases: []Phase{{FromS: 0, ToS: 600, RatePerMin: 3}}, Patience: Patience{MedianMs: 1, MinMs: 1, MaxMs: 1}}
		total += len(s.Schedule())
	}
	if mean := float64(total) / 40; mean < 27 || mean > 33 { // 3/min for 10 min = 30
		t.Fatalf("mean arrivals %.1f, want about 30", mean)
	}
}

func TestSummarizeCountsEachOutcomeOnce(t *testing.T) {
	yes, no := true, false
	recs := []Record{
		{Admitted: &yes, DoneStatus: "ok"}, {Admitted: &no, DoneStatus: "rate_limited"},
		{Admitted: &yes, DoneStatus: "timeout"}, {Admitted: &yes, GaveUp: true}, {Error: "HTTP 503"},
	}
	s := Summarize("x", recs, 60)
	want := Summary{Scenario: "x", Offered: 5, Admitted: 3, Completed: 1, Rejected: 1, TimedOut: 1, GaveUp: 1, Errors: 1,
		DurationS: 60, OfferedPerMin: 5}
	if s != want {
		t.Fatalf("got %+v want %+v", s, want)
	}
}

func fakeAsk(delay time.Duration, status string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: meta\ndata: {\"admitted\": true}\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-time.After(delay):
		case <-r.Context().Done():
			return
		}
		fmt.Fprint(w, "event: content\ndata: {\"text\": \"hi\"}\n\n")
		fmt.Fprintf(w, "event: done\ndata: {\"status\": %q, \"tier\": \"normal\"}\n\n", status)
	}))
}

func TestOneReadsSSEAndRecordsFirstContent(t *testing.T) {
	srv := fakeAsk(10*time.Millisecond, "ok")
	defer srv.Close()
	s := scenario()
	s.Target = srv.URL
	rec := One(srv.Client(), s, Arrival{QuestionID: "q05", Patience: 5 * time.Second}, "q?", time.Now())
	if rec.DoneStatus != "ok" || rec.Admitted == nil || !*rec.Admitted || rec.FirstContentMs == nil || rec.GaveUp {
		t.Fatalf("%+v", rec)
	}
	if *rec.FirstContentMs < 10 || rec.TotalMs < *rec.FirstContentMs {
		t.Fatalf("timings: first %v total %v", *rec.FirstContentMs, rec.TotalMs)
	}
}

func TestUserGivesUpAtPatience(t *testing.T) {
	srv := fakeAsk(2*time.Second, "ok")
	defer srv.Close()
	s := scenario()
	s.Target = srv.URL
	rec := One(srv.Client(), s, Arrival{QuestionID: "q05", Patience: 100 * time.Millisecond}, "q?", time.Now())
	if !rec.GaveUp || rec.FirstContentMs != nil || rec.TotalMs > 1500 {
		t.Fatalf("user must give up at patience: %+v", rec)
	}
}

func TestRunWritesRecordsAndSummary(t *testing.T) {
	srv := fakeAsk(5*time.Millisecond, "ok")
	defer srv.Close()
	dir := t.TempDir()
	s := &Scenario{Name: "mini", Source: "stand-in", Target: srv.URL, QuestionIDs: []string{"q05"}, DurationS: 2, Seed: 3,
		Phases: []Phase{{FromS: 0, ToS: 2, RatePerMin: 600}}, Patience: Patience{MedianMs: 2000, MinMs: 2000, MaxMs: 2000}}
	sum, path, err := Run(s, map[string]string{"q05": "What?"}, dir, t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	if sum.Offered == 0 || sum.Completed != sum.Offered {
		t.Fatalf("%+v", sum)
	}
	b, _ := os.ReadFile(path)
	if strings.Count(string(b), "\n") != sum.Offered {
		t.Fatal("one record per offered request")
	}
	if _, err := os.Stat(strings.TrimSuffix(path, ".jsonl") + ".summary.json"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Run(&Scenario{Name: "bad", Target: srv.URL, QuestionIDs: []string{"q99"}, Phases: s.Phases},
		map[string]string{}, filepath.Join(dir, "x"), t.Logf); err == nil {
		t.Fatal("unknown question ids must fail before any request")
	}
}
