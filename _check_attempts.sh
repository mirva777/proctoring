#!/bin/bash
PGPASSWORD=321456 psql "host=192.168.20.91 port=5433 user=root dbname=moodle sslmode=disable" <<'EOF'
-- Check distinct status values
SELECT DISTINCT status, COUNT(*) as cnt
FROM mdl_quizaccess_proctoring_logs
WHERE webcampicture != '' AND webcampicture IS NOT NULL
GROUP BY status
ORDER BY status;

-- Check the quiz_attempts table
SELECT id, quiz, userid, attempt, state, timestart, timefinish
FROM mdl_quiz_attempts
WHERE quiz = (SELECT DISTINCT quizid FROM mdl_quizaccess_proctoring_logs LIMIT 1)
ORDER BY userid, attempt;

-- Check what the status column maps to
SELECT pl.id, pl.userid, pl.status, pl.timemodified, LEFT(pl.webcampicture, 60)
FROM mdl_quizaccess_proctoring_logs pl
WHERE pl.webcampicture != '' AND pl.webcampicture IS NOT NULL
ORDER BY pl.userid, pl.timemodified
LIMIT 10;
EOF
