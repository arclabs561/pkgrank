package shared

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"golang.org/x/term"
)

var logLevel zerolog.Level
var logOutput io.Writer

func NewLogger() zerolog.Logger {
	return log.Level(logLevel).
		Output(logOutput).
		With().
		// Caller().
		Logger()
}

func SetGlobalLogger() {
	log.Logger = NewLogger()
	zerolog.DefaultContextLogger = &log.Logger
}

func init() {
	initLogOutput()
	initLogFormat()
	initLogLevel()

	SetGlobalLogger()
}

func initLogOutput() {
	logOutputRaw := os.Getenv("LOG_OUTPUT")
	if logOutputRaw == "" {
		logOutput = os.Stderr
		return
	}

	dir := filepath.Dir(logOutputRaw)
	if err := os.MkdirAll(dir, 0755); err != nil {
		panic(fmt.Sprintf("unable to create directory for log output %q: %v", dir, err))
	}

	file, err := os.OpenFile(logOutputRaw, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(fmt.Sprintf("unable to open log output file %q: %v", logOutputRaw, err))
	}

	logOutput = file
}

func initLogFormat() {
	logFormatRaw := os.Getenv("LOG_FORMAT")
	if logFormatRaw == "" {
		if term.IsTerminal(int(os.Stdout.Fd())) {
			logFormatRaw = "console"
		} else {
			logFormatRaw = "json"
		}
	}

	switch {
	case strings.EqualFold(logFormatRaw, "console"):
		logOutput = zerolog.ConsoleWriter{Out: logOutput}
	case strings.EqualFold(logFormatRaw, "json"):
		// LogOutput remains the same for JSON
	default:
		panic(fmt.Sprintf("invalid log format %q", logFormatRaw))
	}
}

func initLogLevel() {
	logLevelRaw := os.Getenv("LOG_LEVEL")
	if logLevelRaw == "" {
		logLevelRaw = "disabled"
	}
	lvl, err := zerolog.ParseLevel(logLevelRaw)
	if err != nil {
		panic(fmt.Sprintf("invalid log level %q: %v", logLevelRaw, err))
	}
	logLevel = lvl
}
