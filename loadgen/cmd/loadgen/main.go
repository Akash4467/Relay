// Command loadgen runs one load scenario against the /ask front door.
//
//	go run ./cmd/loadgen -scenario ../config/scenarios/steady.json -root ..
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"path/filepath"

	"relay/loadgen/internal/run"
)

func main() {
	scenarioPath := flag.String("scenario", "", "scenario JSON")
	root := flag.String("root", "..", "project root (questions_file and the output dir are relative to it)")
	target := flag.String("target", "", "override the scenario's /ask URL")
	flag.Parse()
	s, err := run.LoadScenario(*scenarioPath)
	if err != nil {
		log.Fatal(err)
	}
	if *target != "" {
		s.Target = *target
	}
	questions, err := run.LoadQuestions(filepath.Join(*root, s.QuestionsFile))
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("scenario %s: %d arrivals scheduled over %.0fs", s.Name, len(s.Schedule()), s.DurationS)
	sum, path, err := run.Run(s, questions, filepath.Join(*root, "results/raw/loadgen"), log.Printf)
	if err != nil {
		log.Fatal(err)
	}
	b, _ := json.MarshalIndent(sum, "", "  ")
	fmt.Printf("%s\nrecords: %s\n", b, path)
}
