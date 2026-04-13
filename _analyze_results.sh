#!/bin/bash
cd /mnt/c/Users/Mirvohid/Desktop/proctoring

echo "=== IMAGE LEVEL RESULTS HEADER ==="
head -1 real_moodle_export/results/image_level_results.csv

echo ""
echo "=== RISK SCORE DISTRIBUTION ==="
awk -F, 'NR>1{score=$14; if(score+0>=50) print "50+"; else if(score+0>=30) print "30-49"; else if(score+0>=10) print "10-29"; else print "0-9"}' real_moodle_export/results/image_level_results.csv | sort | uniq -c | sort -rn

echo ""
echo "=== OVERALL RISK LEVELS (student summaries) ==="
awk -F, 'NR>1{print $13}' real_moodle_export/results/student_summary.csv | sort | uniq -c | sort -rn

echo ""
echo "=== TOP DETECTION REASONS ==="
grep -oP '"[A-Z_]+"' real_moodle_export/results/student_summary.csv | sort | uniq -c | sort -rn | head -15

echo ""
echo "=== SAMPLE HIGH-RISK FRAMES ==="
awk -F, 'NR>1 && $14+0 >= 40' real_moodle_export/results/image_level_results.csv | head -5

echo ""
echo "=== SAMPLE LOW-RISK FRAMES ==="
awk -F, 'NR>1 && $14+0 == 0' real_moodle_export/results/image_level_results.csv | head -5

echo ""
echo "=== STUDENT SUMMARY (first 10 rows) ==="
head -11 real_moodle_export/results/student_summary.csv | column -s, -t 2>/dev/null || head -11 real_moodle_export/results/student_summary.csv
