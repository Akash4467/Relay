package sim

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Config mirrors config/standin.json.
type Config struct {
	Seed   uint64 `json:"seed"`
	Models map[string]struct {
		RPM float64 `json:"rpm"`
		RPD float64 `json:"rpd"`
		TPM float64 `json:"tpm"`
		TPD float64 `json:"tpd"`
	} `json:"models"`
	Latency struct {
		FirstByteLogMu    float64 `json:"first_byte_log_mu"`
		FirstByteLogSigma float64 `json:"first_byte_log_sigma"`
		ReasoningTPS      float64 `json:"reasoning_tokens_per_s"`
		ContentTPS        float64 `json:"content_tokens_per_s"`
		ChunkTokens       int     `json:"chunk_tokens"`
		LoadFirstByte     float64 `json:"load_first_byte_per_inflight"`
		LoadRate          float64 `json:"load_rate_per_inflight"`
	} `json:"latency"`
	RequestOverheadTokens float64 `json:"request_overhead_tokens"`
	CharsPerToken         float64 `json:"chars_per_token"`
	Faults                Faults  `json:"faults"`
	FixturesDir           string  `json:"fixtures_dir"`
	LogDir                string  `json:"log_dir"`
}

type Faults struct {
	P500       float64 `json:"p_500"`
	PSlow      float64 `json:"p_slow_success"`
	SlowFactor float64 `json:"slow_factor"`
	PStall     float64 `json:"p_stall"`
	StallMs    float64 `json:"stall_ms"`
	PTimeout   float64 `json:"p_timeout"`
}

func LoadConfig(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if c.Latency.ChunkTokens <= 0 || c.Latency.ContentTPS <= 0 || c.Latency.ReasoningTPS <= 0 || c.CharsPerToken <= 0 {
		return nil, fmt.Errorf("%s: latency rates, chunk_tokens and chars_per_token must be positive", path)
	}
	return &c, nil
}

// Draft is one recorded real answer with its real token counts.
type Draft struct {
	Content          string  `json:"content"`
	CompletionTokens float64 `json:"completion_tokens"`
	ReasoningTokens  float64 `json:"reasoning_tokens"`
}

// Fixture: Mode A performance replies, recorded from real calls (drafts[0] = first answer, drafts[1] = redraft).
type Fixture struct {
	QuestionID    string  `json:"question_id"`
	Model         string  `json:"model"`
	PromptVersion string  `json:"prompt_version"`
	RecordedAt    string  `json:"recorded_at"`
	Drafts        []Draft `json:"drafts"`
}

func LoadFixtures(dir string) (map[string]Fixture, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	out := map[string]Fixture{}
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		var fx Fixture
		if err := json.Unmarshal(b, &fx); err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		if fx.QuestionID == "" || len(fx.Drafts) == 0 {
			return nil, fmt.Errorf("%s: question_id and at least one draft are required", f)
		}
		out[fx.QuestionID] = fx
	}
	return out, nil
}

// Pick returns the draft for this attempt: a redraft request gets the recorded redraft when there is one.
func (f Fixture) Pick(redraft bool) Draft {
	if redraft && len(f.Drafts) > 1 {
		return f.Drafts[1]
	}
	return f.Drafts[0]
}

// Plan is everything decided for one request before it is served.
type Plan struct {
	Fault      string // "", "500", "slow", "stall", "timeout"
	FirstByte  time.Duration
	Reasoning  time.Duration
	ChunkGap   time.Duration
	StallAfter int // chunk index after which the stream pauses
	Stall      time.Duration
}

// Planner samples latencies and faults from a seeded generator, so a scenario run is reproducible.
type Planner struct {
	mu  sync.Mutex
	rng *rand.Rand
	cfg *Config
}

func NewPlanner(cfg *Config) *Planner {
	return &Planner{rng: rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0x9e3779b97f4a7c15)), cfg: cfg}
}

// Plan: first byte ~ LogNormal(mu, sigma) * (1 + a*inflight); hidden reasoning at reasoning_tps and content at
// content_tps, both slowed by (1 + c*inflight).
func (p *Planner) Plan(inflight int, reasoningTokens, contentTokens float64) Plan {
	p.mu.Lock()
	defer p.mu.Unlock()
	l, f := p.cfg.Latency, p.cfg.Faults
	n := float64(inflight)
	firstByte := math.Exp(l.FirstByteLogMu+l.FirstByteLogSigma*p.rng.NormFloat64()) * (1 + l.LoadFirstByte*n)
	slow := 1 + l.LoadRate*n
	plan := Plan{
		FirstByte: ms(firstByte),
		Reasoning: ms(reasoningTokens / l.ReasoningTPS * 1000 * slow),
		ChunkGap:  ms(float64(l.ChunkTokens) / l.ContentTPS * 1000 * slow),
	}
	switch u := p.rng.Float64(); {
	case u < f.P500:
		plan.Fault = "500"
	case u < f.P500+f.PTimeout:
		plan.Fault = "timeout"
	case u < f.P500+f.PTimeout+f.PSlow:
		plan.Fault = "slow"
		plan.FirstByte = ms(firstByte * f.SlowFactor)
	case u < f.P500+f.PTimeout+f.PSlow+f.PStall:
		plan.Fault = "stall"
		chunks := int(math.Max(1, math.Ceil(contentTokens/float64(l.ChunkTokens))))
		plan.StallAfter = p.rng.IntN(chunks)
		plan.Stall = ms(f.StallMs)
	}
	return plan
}

func ms(v float64) time.Duration { return time.Duration(v * float64(time.Millisecond)) }

// PromptTokens estimates a request's prompt tokens like the provider counts them: content length / chars per
// token, plus the fixed per-request overhead measured on gpt-oss (67 tokens).
func (c *Config) PromptTokens(messages []map[string]any) float64 {
	chars := 0
	for _, m := range messages {
		if s, ok := m["content"].(string); ok {
			chars += len([]rune(s))
		}
	}
	return math.Ceil(float64(chars)/c.CharsPerToken) + c.RequestOverheadTokens
}

// IsRedraft: the app's redraft prompt ends with this user message (relay/llm/prompts.py redraft_messages).
func IsRedraft(messages []map[string]any) bool {
	if len(messages) == 0 {
		return false
	}
	s, _ := messages[len(messages)-1]["content"].(string)
	return strings.HasPrefix(s, "Your previous answer had unsupported sentences")
}
