// Command standin runs the stand-in model service.
//
//	go run ./cmd/standin -config ../config/standin.json -port 8001 -root ..
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"time"

	"relay/standin/internal/sim"
)

func main() {
	configPath := flag.String("config", "../config/standin.json", "stand-in config (JSON)")
	port := flag.Int("port", 8001, "listen port")
	root := flag.String("root", "..", "project root; fixtures_dir and log_dir in the config are relative to it")
	flag.Parse()

	cfg, err := sim.LoadConfig(*configPath)
	if err != nil {
		log.Fatal(err)
	}
	fixtures, err := sim.LoadFixtures(filepath.Join(*root, cfg.FixturesDir))
	if err != nil {
		log.Fatal(err)
	}
	if len(fixtures) == 0 {
		log.Fatalf("no fixtures in %s: record real answers first", cfg.FixturesDir)
	}
	logPath := filepath.Join(*root, cfg.LogDir, time.Now().UTC().Format("2006-01-02T15-04-05")+".jsonl")
	srv, err := sim.NewServer(cfg, fixtures, logPath)
	if err != nil {
		log.Fatal(err)
	}
	addr := fmt.Sprintf("127.0.0.1:%d", *port)
	log.Printf("stand-in on http://%s/v1  fixtures=%d  log=%s", addr, len(fixtures), logPath)
	log.Fatal(http.ListenAndServe(addr, srv.Handler()))
}
