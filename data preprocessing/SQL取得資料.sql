SELECT n.hadm_id,
       LEFT(n.text, 2000) AS note_text,
       string_agg(d.icd_code, ',' ORDER BY d.icd_code) AS icd_codes
FROM mimiciv_note.discharge n
INNER JOIN mimic_hosp.diagnoses_icd d
    ON n.hadm_id = d.hadm_id
WHERE d.icd_version = 10
GROUP BY n.hadm_id, n.text
ORDER BY n.hadm_id
LIMIT 4000;


