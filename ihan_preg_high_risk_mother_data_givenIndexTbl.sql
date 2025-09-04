------ This query will assume that code-index dictionary tables are given through parameters: 
--CREATION of MEDICAL DX CODE TABLE 
DROP TABLE IF EXISTS &dx_t1;
CREATE TABLE &dx_t1 AS
(
SELECT DISTINCT
SBSCRBR_KEY AS MCID
,DOS
,TYPE as cd_type
,UPPER(TRIM(VALUE)) as diag_cd
FROM &input_tbl
WHERE TYPE ='DIAG_CD'
);

select * from &dx_t1 limit 5;
select count(*) from &dx_t1;


--CREATION of PROCEDURE CODE TABLE 
DROP TABLE IF EXISTS &proc_t1;
CREATE TABLE &proc_t1 AS
(
SELECT DISTINCT
SBSCRBR_KEY AS MCID
,DOS
,TYPE as cd_type
,UPPER(TRIM(VALUE)) as proc_cd
FROM &input_tbl
WHERE TYPE ='HLTH_SRVC_CD'
);

select * from &proc_t1 limit 5;
select count(*) from &proc_t1;

--CREATION of REVENUE CODE TABLE 
DROP TABLE IF EXISTS &rvnu_t1;
CREATE TABLE &rvnu_t1 AS
(
SELECT DISTINCT
SBSCRBR_KEY AS MCID
,DOS
,TYPE as cd_type
,UPPER(TRIM(VALUE)) as rvnu_cd
FROM &input_tbl
WHERE TYPE ='RVNU_CD'
);

select * from &rvnu_t1 limit 5;
select count(*) from &rvnu_t1;


--CREATION of MEDICAL RX(GPI) CODE TABLE 
DROP TABLE IF EXISTS &gpi_t1;
CREATE TABLE &gpi_t1 AS
(
SELECT DISTINCT
SBSCRBR_KEY AS MCID
,DOS
,TYPE as cd_type
,UPPER(TRIM(VALUE)) as gpi_cd
FROM &input_tbl
WHERE TYPE ='GPI_02_GRP_CD'
);

select * from &gpi_t1 limit 5;
select count(*) from &gpi_t1;

-------- data index and formatting --------
-- index table (given): ihan_final_idx
-- merging diag code and index of the code 
DROP TABLE IF EXISTS &dx_t3;

CREATE TABLE &dx_t3 AS
SELECT mcid,
CONCAT('\'', dos, '\'') dos,
CONCAT('\'',d.diag_cd, '\'') AS diag_cd,
idx AS nidx
,ROW_NUMBER() OVER (PARTITION BY mcid, dos ORDER BY d.diag_cd) rid
FROM (&dx_t1) d
JOIN (&ihan_final_idx) i
WHERE i.cd_type = d.cd_type and d.diag_cd = i.cd_value;

SELECT * FROM (&dx_t3) LIMIT 10;

select count(*) from (&dx_t3);

SELECT * FROM (&dx_t3) WHERE nidx IS NULL;


--merging procedure code and procedure code index 
DROP TABLE IF EXISTS &proc_t3;

CREATE TABLE &proc_t3 AS
SELECT mcid,
CONCAT('\'', dos, '\'') dos,
CONCAT('\'', p.proc_cd, '\'') proc_cd,
idx AS nidx
,ROW_NUMBER() OVER (PARTITION BY mcid, dos ORDER BY p.proc_cd) rid
FROM (&proc_t1) p
JOIN (&ihan_final_idx) i
WHERE p.proc_cd = i.cd_value
AND p.cd_type = i.cd_type;

SELECT * FROM (&proc_t3) LIMIT 30;
select count(*) from (&proc_t3);


-- merging rvnu code and index of the code 
DROP TABLE IF EXISTS &rvnu_t3;

CREATE TABLE &rvnu_t3 AS
SELECT mcid,
CONCAT('\'', dos, '\'') dos,
CONCAT('\'', r.rvnu_cd, '\'') AS rvnu_cd,
idx AS nidx
,ROW_NUMBER() OVER (PARTITION BY mcid, dos ORDER BY r.rvnu_cd) rid
FROM (&rvnu_t1) r
JOIN (&ihan_final_idx) i
WHERE i.cd_type = r.cd_type and r.rvnu_cd = i.cd_value;

SELECT * FROM (&rvnu_t3) LIMIT 10;

select count(*) from (&rvnu_t3);

SELECT * FROM (&rvnu_t3) WHERE nidx IS NULL;


-- merging GPI code and index of the code 
DROP TABLE IF EXISTS &gpi_t3;

CREATE TABLE &gpi_t3 AS
SELECT mcid,
CONCAT('\'', dos, '\'') dos,
CONCAT('\'', g.gpi_cd, '\'') AS gpi_cd,
idx AS nidx
,ROW_NUMBER() OVER (PARTITION BY mcid, dos ORDER BY g.gpi_cd) rid
FROM (&gpi_t1) g
JOIN (&ihan_final_idx) i
WHERE i.cd_type = g.cd_type and g.gpi_cd = i.cd_value;

SELECT * FROM (&gpi_t3) LIMIT 10;

select count(*) from (&gpi_t3);

SELECT * FROM (&gpi_t3) WHERE nidx IS NULL;

--displaying rows from tables
SELECT COUNT(*) FROM (&dx_t3);

SELECT COUNT(*) FROM (&proc_t3);

SELECT COUNT(*) FROM (&rvnu_t3);

SELECT COUNT(*) FROM (&gpi_t3);

--creating final medical dx code table 
DROP TABLE IF EXISTS &final_dx_data;

CREATE TABLE &final_dx_data AS
SELECT mcid,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(dos), ','), ']') dos,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(dx_list), ','), ']') dx_code_list,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(idx_list), ','), ']') idx_list,
COUNT(dos) vst_cnt
FROM
(
SELECT mcid, dos,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(diag_cd), ','), ']') dx_list,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(TO_VARCHAR(nidx)), ','), ']') idx_list
FROM
(
SELECT mcid, dos, diag_cd, nidx
FROM (&dx_t3)
WHERE rid <= &max_num_code_per_visit
) A
GROUP BY mcid, dos
ORDER BY mcid, dos
) B
GROUP BY mcid;

SELECT * FROM (&final_dx_data) LIMIT 10;

--creating final medical procedure code table 
DROP TABLE IF EXISTS &final_proc_data;

CREATE TABLE &final_proc_data AS
SELECT mcid,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(dos), ','), ']') dos,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(proc_list), ','), ']') px_code_list,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(idx_list), ','), ']') idx_list
FROM
(
SELECT mcid, dos,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(proc_cd), ','), ']') proc_list,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(TO_VARCHAR(nidx)), ','), ']') idx_list
FROM (
SELECT mcid, dos, proc_cd, nidx
FROM (&proc_t3)
WHERE rid <= &max_num_code_per_visit
) A
GROUP BY mcid, dos
ORDER BY mcid, dos
) B
GROUP BY mcid;

SELECT * FROM (&final_proc_data) LIMIT 10;


--creating final medical rvnu code table 
DROP TABLE IF EXISTS &final_rvnu_data;

CREATE TABLE &final_rvnu_data AS
SELECT mcid,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(dos), ','), ']') dos,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(rvnu_list), ','), ']') rvnu_code_list,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(idx_list), ','), ']') idx_list
FROM
(
SELECT mcid, dos,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(rvnu_cd), ','), ']') rvnu_list,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(TO_VARCHAR(nidx)), ','), ']') idx_list
FROM (
SELECT mcid, dos, rvnu_cd, nidx
FROM (&rvnu_t3)
WHERE rid <= &max_num_code_per_visit
) A
GROUP BY mcid, dos
ORDER BY mcid, dos
) B
GROUP BY mcid;

SELECT * FROM (&final_rvnu_data) LIMIT 10;

--creating final medical gpi code table 
DROP TABLE IF EXISTS &final_gpi_data;

CREATE TABLE &final_gpi_data AS
SELECT mcid,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(dos), ','), ']') dos,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(gpi_list), ','), ']') gpi_code_list,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(idx_list), ','), ']') idx_list
FROM
(
SELECT mcid, dos,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(gpi_cd), ','), ']') gpi_list,
CONCAT('[', ARRAY_TO_STRING(ARRAY_AGG(TO_VARCHAR(nidx)), ','), ']') idx_list
FROM (
SELECT mcid, dos, gpi_cd, nidx
FROM (&gpi_t3)
WHERE rid <= &max_num_code_per_visit
) A
GROUP BY mcid, dos
ORDER BY mcid, dos
) B
GROUP BY mcid;

SELECT * FROM (&final_gpi_data) LIMIT 10;

select count(*) from (&final_dx_data);
select count(*) from (&final_proc_data);
select count(*) from (&final_rvnu_data);
select count(*) from (&final_gpi_data);


-----------creating static features and target feature table later, will be merged with medical code tables----------
DROP TABLE IF EXISTS &static_tbl;
CREATE TABLE &static_tbl as 
SELECT
    SBSCRBR_KEY AS mcid,
    MIN(baby_brth_dt) as baby_birth_dt,
    MIN(MOM_AGE) as mom_age,  
    MIN(TARGET) as target,      
    CAST(MIN(CASE WHEN TYPE = 'high_risk_proc_cd_cnt' THEN VALUE END) AS INTEGER) AS high_risk_proc_cd_cnt,
    CAST(MIN(CASE WHEN TYPE = 'TRANSVAGINAL_SONOGRAMS' THEN VALUE END) AS INTEGER) AS transvaginal_sonograms,
    CAST(MIN(CASE WHEN TYPE = 'VASA_PRIVIA' THEN VALUE END) AS INTEGER) AS vasa_privia,
    CAST(MIN(CASE WHEN TYPE = 'ANATOMICAL_FETAL_SURVEY' THEN VALUE END) AS INTEGER) AS anatomical_fetal_survey,
    CAST(MIN(CASE WHEN TYPE = 'MLTPL_GEST' THEN VALUE END) AS INTEGER) AS mltpl_gest,
    CAST(MIN(CASE WHEN TYPE = 'EPILEPSY' THEN VALUE END) AS INTEGER) AS epilepsy,
    CAST(MIN(CASE WHEN TYPE = 'SUM_INPATIENT_VISIT' THEN VALUE END) AS INTEGER) AS sum_inpatient_visit,
    CAST(MIN(CASE WHEN TYPE = 'BLOOD_PRESR' THEN VALUE END) AS INTEGER) AS blood_presr,
    CAST(MIN(CASE WHEN TYPE = 'NT_TEST' THEN VALUE END) AS INTEGER) AS nt_test,
    CAST(MIN(CASE WHEN TYPE = 'COMPL_SMKNG_ALCHL' THEN VALUE END) AS INTEGER) AS compl_smkng_alchl,
    CAST(MIN(CASE WHEN TYPE = 'TRANSVAGINAL_ULTRASOUND_EXAM' THEN VALUE END) AS INTEGER) AS transvaginal_ultrasound_exam,
    CAST(MIN(CASE WHEN TYPE = 'NST_TEST' THEN VALUE END) AS INTEGER) AS nst_test,
    CAST(MIN(CASE WHEN TYPE = 'SUM_EMERGENCY_VISIT' THEN VALUE END) AS INTEGER) AS sum_emergency_visit,
    CAST(MIN(CASE WHEN TYPE = 'paid_9_months' THEN VALUE END) AS FLOAT) AS paid_9_months,
    CAST(MIN(CASE WHEN TYPE = 'DIABETES' THEN VALUE END) AS INTEGER) AS diabetes,
    CAST(MIN(CASE WHEN TYPE = 'INFERTILITY' THEN VALUE END) AS INTEGER) AS infertility,
    CAST(MIN(CASE WHEN TYPE = 'PREECLAMPSIA' THEN VALUE END) AS INTEGER) AS preeclampsia,
    CAST(MIN(CASE WHEN TYPE = 'PAPP' THEN VALUE END) AS INTEGER) AS papp,
    CAST(MIN(CASE WHEN TYPE = 'high_risk_diag_cnt' THEN VALUE END) AS INTEGER) AS high_risk_diag_cnt,
    CAST(MIN(CASE WHEN TYPE = 'DOPPLER_FLOW_STUDIES' THEN VALUE END) AS INTEGER) AS doppler_flow_studies,
    CAST(MIN(CASE WHEN TYPE = 'IVF' THEN VALUE END) AS INTEGER) AS ivf,
    CAST(MIN(CASE WHEN TYPE = 'COMPLICATIONS_MLTPL_GEST' THEN VALUE END) AS INTEGER) AS complications_mltpl_gest,
    CAST(MIN(CASE WHEN TYPE = 'TRANSCERVICAL_AND_TRANSABDOMINAL_CHORIONIC_VILLUS_SAMPLING' THEN VALUE END) AS INTEGER) AS transcervical_and_transabdominal_chorionic_villus_sampling,
    CAST(MIN(CASE WHEN TYPE = 'FETAL_ECHOCARDIOGRAPHY' THEN VALUE END) AS INTEGER) AS fetal_echocardiography,
    CAST(MIN(CASE WHEN TYPE = 'ENDOMETROSIS' THEN VALUE END) AS INTEGER) AS endometrosis,
    CAST(MIN(CASE WHEN TYPE = 'PCOS' THEN VALUE END) AS INTEGER) AS pcos,
    CAST(MIN(CASE WHEN TYPE = 'AMNIOCENTESIS' THEN VALUE END) AS INTEGER) AS amniocentesis,
    CAST(MIN(CASE WHEN TYPE = 'SUM_OUTPATIENT_VISIT' THEN VALUE END) AS INTEGER) AS sum_outpatient_visit,
    CAST(MIN(CASE WHEN TYPE = 'GEST_DIABETES' THEN VALUE END) AS INTEGER) AS gest_diabetes,
    CAST(MIN(CASE WHEN TYPE = 'HCG_SCREENING' THEN VALUE END) AS INTEGER) AS hcg_screening,
    CAST(MIN(CASE WHEN TYPE = 'paid_12_months' THEN VALUE END) AS FLOAT) AS paid_12_months,
    CAST(MIN(CASE WHEN TYPE = 'paid_6_months' THEN VALUE END) AS FLOAT) AS paid_6_months,
    CAST(MIN(CASE WHEN TYPE = 'CANCER_HIST_PRSNL' THEN VALUE END) AS INTEGER) AS cancer_hist_prsnl,
    CAST(MIN(CASE WHEN TYPE = 'AFI_INDEX' THEN VALUE END) AS INTEGER) AS afi_index,
    CAST(MIN(CASE WHEN TYPE = 'GROWTH_RESTRICTION' THEN VALUE END) AS INTEGER) AS growth_restriction,
    CAST(MIN(CASE WHEN TYPE = 'OBESITY_HIGH_BMI' THEN VALUE END) AS INTEGER) AS obesity_high_bmi,
    CAST(MIN(CASE WHEN TYPE = 'INFERTILITY_MEDICINE' THEN VALUE END) AS INTEGER) AS infertility_medicine,
    CAST(MIN(CASE WHEN TYPE = 'CANCER_HIST_FAMLY' THEN VALUE END) AS INTEGER) AS cancer_hist_famly,
    CAST(MIN(CASE WHEN TYPE = 'BPP_PROFILE' THEN VALUE END) AS INTEGER) AS bpp_profile,
    CAST(MIN(CASE WHEN TYPE = 'paid_3_months' THEN VALUE END) AS FLOAT) AS paid_3_months
FROM &input_tbl
GROUP BY SBSCRBR_KEY;
    

select * from &static_tbl limit 10;
select count(*) from &static_tbl;
SELECT target,COUNT(mcid) from &static_tbl GROUP BY target;


--------------------- MERGE MEDICAL CODES (Diag/PROC/lab/RX), code pair (labpair) and stars feature WITH TARGET TABLE 
drop table if exists &ihan_tbl&iter;

create table &ihan_tbl&iter as 
select t1.*,
COALESCE(t2.dos,'[]') as dos1,
COALESCE(t2.idx_list,'[]') as datlist1,
COALESCE(t3.dos,'[]') as dos2,
COALESCE(t3.idx_list,'[]') as datlist2,
COALESCE(t4.dos,'[]') as dos3,
COALESCE(t4.idx_list,'[]') as datlist3,
COALESCE(t5.dos,'[]') as dos4,
COALESCE(t5.idx_list,'[]') as datlist4 
from (&static_tbl) as t1
left join 
(&final_dx_data) as t2
on t1.mcid =t2.mcid
left join 
(&final_proc_data) as t3
on t1.mcid =t3.mcid
left join 
(&final_rvnu_data) as t4
on t1.mcid =t4.mcid
left join
(&final_gpi_data) as t5
on t1.mcid =t5.mcid;


select count(*) from &ihan_tbl&iter;
select * from &ihan_tbl&iter limit 5;
select target,COUNT(mcid) from &ihan_tbl&iter GROUP BY target;


------------ cleanup
drop table IF EXISTS &dx_t1;
drop table IF EXISTS &proc_t1;
drop table IF EXISTS &rvnu_t1;
drop table IF EXISTS &gpi_t1;
drop table IF EXISTS &dx_t3;
drop table IF EXISTS &proc_t3;
drop table IF EXISTS &rvnu_t3;
drop table IF EXISTS &gpi_t3;
drop table IF EXISTS &final_dx_data;
drop table IF EXISTS &final_proc_data;
drop table IF EXISTS &final_rvnu_data;
drop table IF EXISTS &final_gpi_data;
drop table IF EXISTS &static_tbl;





