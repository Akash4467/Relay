// Package sim is the stand-in model service: Groq-like limits, latency, faults and fixtures (DESIGN §16).
package sim

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// Bucket refills continuously at Cap/Period per second: the real per-day request limit was observed to refill
// one request every 86.4 s (1,000 per day), not at a fixed daily reset.
type Bucket struct {
	Cap     float64
	Period  time.Duration
	level   float64
	updated time.Time
}

func NewBucket(capacity float64, period time.Duration, now time.Time) *Bucket {
	return &Bucket{Cap: capacity, Period: period, level: capacity, updated: now}
}

func (b *Bucket) refill(now time.Time) {
	if now.After(b.updated) {
		b.level = math.Min(b.Cap, b.level+now.Sub(b.updated).Seconds()*b.Cap/b.Period.Seconds())
		b.updated = now
	}
}

func (b *Bucket) Available(now time.Time) float64 { b.refill(now); return b.level }

// WaitFor is the time until n units are available.
func (b *Bucket) WaitFor(n float64, now time.Time) time.Duration {
	b.refill(now)
	if n <= b.level {
		return 0
	}
	return time.Duration((n - b.level) * b.Period.Seconds() / b.Cap * float64(time.Second))
}

func (b *Bucket) Take(n float64, now time.Time) { b.refill(now); b.level -= n }

// Scale changes the capacity (e.g. 0.5 for "per-minute limit halved"), never leaving more than the new capacity.
func (b *Bucket) Scale(capacity float64, now time.Time) {
	b.refill(now)
	b.Cap = capacity
	b.level = math.Min(b.level, capacity)
}

// Empty sets the level to zero ("daily budget exhausted"); it then refills at the normal rate.
func (b *Bucket) Empty(now time.Time) { b.refill(now); b.level = 0 }

// Limits holds one model's four buckets. A request needs all four at once.
type Limits struct {
	mu                 sync.Mutex
	RPM, RPD, TPM, TPD *Bucket
	base               map[string]float64
}

func NewLimits(rpm, rpd, tpm, tpd float64, now time.Time) *Limits {
	return &Limits{
		RPM: NewBucket(rpm, time.Minute, now), RPD: NewBucket(rpd, 24*time.Hour, now),
		TPM: NewBucket(tpm, time.Minute, now), TPD: NewBucket(tpd, 24*time.Hour, now),
		base: map[string]float64{"rpm": rpm, "rpd": rpd, "tpm": tpm, "tpd": tpd},
	}
}

// Rejection says which limit refused a request and how long until it would fit.
type Rejection struct {
	Limit      string // "requests per minute (RPM)" etc., as in Groq's error message
	Code       string // "requests" | "tokens"
	Cap, Used  float64
	Requested  float64
	RetryAfter time.Duration
	PerDay     bool
}

// Admit takes 1 request and `tokens` tokens from every bucket, or nothing and returns the binding limit.
func (l *Limits) Admit(tokens float64, now time.Time) *Rejection {
	l.mu.Lock()
	defer l.mu.Unlock()
	checks := []struct {
		b      *Bucket
		n      float64
		name   string
		code   string
		perDay bool
	}{
		{l.RPD, 1, "requests per day (RPD)", "requests", true},
		{l.TPD, tokens, "tokens per day (TPD)", "tokens", true},
		{l.RPM, 1, "requests per minute (RPM)", "requests", false},
		{l.TPM, tokens, "tokens per minute (TPM)", "tokens", false},
	}
	for _, c := range checks {
		if w := c.b.WaitFor(c.n, now); w > 0 {
			retry := w
			if c.n > c.b.Cap { // can never fit: tell the client to come back after a full period
				retry = c.b.Period
			}
			return &Rejection{Limit: c.name, Code: c.code, Cap: c.b.Cap, Used: c.b.Cap - c.b.Available(now),
				Requested: c.n, RetryAfter: retry, PerDay: c.perDay}
		}
	}
	for _, c := range checks {
		c.b.Take(c.n, now)
	}
	return nil
}

// Headers returns Groq's rate-limit headers: *-requests are per DAY, *-tokens are per MINUTE.
func (l *Limits) Headers(now time.Time) map[string]string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return map[string]string{
		"x-ratelimit-limit-requests":     fmt.Sprintf("%.0f", l.RPD.Cap),
		"x-ratelimit-remaining-requests": fmt.Sprintf("%.0f", math.Max(0, math.Floor(l.RPD.Available(now)))),
		"x-ratelimit-reset-requests":     GroqDuration(l.RPD.WaitFor(l.RPD.Cap, now)),
		"x-ratelimit-limit-tokens":       fmt.Sprintf("%.0f", l.TPM.Cap),
		"x-ratelimit-remaining-tokens":   fmt.Sprintf("%.0f", math.Max(0, math.Floor(l.TPM.Available(now)))),
		"x-ratelimit-reset-tokens":       GroqDuration(l.TPM.WaitFor(l.TPM.Cap, now)),
	}
}

// Configure applies an admin change: scale per-minute limits, empty the daily budget, or restore the base.
func (l *Limits) Configure(rpmScale, tpmScale float64, exhaustDaily, reset bool, now time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if reset {
		l.RPM, l.RPD = NewBucket(l.base["rpm"], time.Minute, now), NewBucket(l.base["rpd"], 24*time.Hour, now)
		l.TPM, l.TPD = NewBucket(l.base["tpm"], time.Minute, now), NewBucket(l.base["tpd"], 24*time.Hour, now)
	}
	if rpmScale > 0 {
		l.RPM.Scale(l.base["rpm"]*rpmScale, now)
	}
	if tpmScale > 0 {
		l.TPM.Scale(l.base["tpm"]*tpmScale, now)
	}
	if exhaustDaily {
		l.RPD.Empty(now)
		l.TPD.Empty(now)
	}
}

// GroqDuration formats like Groq's reset headers: "705ms", "7.66s", "1m26.4s".
func GroqDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	if d < time.Minute {
		return trim(fmt.Sprintf("%.2f", d.Seconds())) + "s"
	}
	m := int(d.Minutes())
	return fmt.Sprintf("%dm%ss", m, trim(fmt.Sprintf("%.1f", d.Seconds()-float64(m*60))))
}

func trim(s string) string {
	for len(s) > 1 && s[len(s)-1] == '0' && containsDot(s) {
		s = s[:len(s)-1]
	}
	if s[len(s)-1] == '.' {
		s = s[:len(s)-1]
	}
	return s
}

func containsDot(s string) bool {
	for _, c := range s {
		if c == '.' {
			return true
		}
	}
	return false
}
