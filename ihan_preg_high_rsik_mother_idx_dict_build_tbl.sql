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
--creation of medical code index for dx table 
DROP TABLE IF EXISTS &dx_t1_idx;

CREATE TABLE &dx_t1_idx AS
SELECT diag_cd,cd_type,
ROW_NUMBER() OVER (ORDER BY cd_type, diag_cd) idx
FROM (
    SELECT DISTINCT diag_cd,cd_type
    FROM (&dx_t1)
) A 
WHERE diag_cd NOT IN ('', 'UNK', 'NA', 'NaN', 'NAN');


SELECT * FROM &dx_t1_idx LIMIT 10;
select count(*) from &dx_t1_idx;

-- fetch min and maximum code index for dx table 
SELECT MIN(idx), MAX(idx) FROM &dx_t1_idx;

--creating code index for the procedure table 
DROP TABLE IF EXISTS &proc_t1_idx;

CREATE TABLE &proc_t1_idx AS
SELECT 
    proc_cd,
    cd_type,
    ROW_NUMBER() OVER (ORDER BY cd_type, proc_cd) idx
FROM (
    SELECT DISTINCT proc_cd,cd_type
    FROM (&proc_t1)
) A 
WHERE proc_cd NOT IN ('', 'UNK', 'NA', 'NaN', 'NAN');

--find out the minimum and maximum index for the procedure table
SELECT * FROM (&proc_t1_idx) LIMIT 10;

SELECT MIN(idx), MAX(idx)
FROM (&proc_t1_idx);

select count(*) from (&proc_t1_idx);

--creating code index for the revenue table 
DROP TABLE IF EXISTS &rvnu_t1_idx;

CREATE TABLE &rvnu_t1_idx AS
SELECT 
    rvnu_cd,
    cd_type,
    ROW_NUMBER() OVER (ORDER BY cd_type, rvnu_cd) idx
FROM (
    SELECT DISTINCT rvnu_cd,cd_type
    FROM (&rvnu_t1)
) A 
WHERE rvnu_cd NOT IN ('', 'UNK', 'NA', 'NaN', 'NAN');

--find out the minimum and maximum index for the revenue table
SELECT * FROM (&rvnu_t1_idx) LIMIT 10;

SELECT MIN(idx), MAX(idx)
FROM (&rvnu_t1_idx);

select count(*) from (&rvnu_t1_idx);

--creating code index for the GPI table 
DROP TABLE IF EXISTS &gpi_t1_idx;

CREATE TABLE &gpi_t1_idx AS
SELECT 
    gpi_cd,
    cd_type,
    ROW_NUMBER() OVER (ORDER BY cd_type, gpi_cd) idx
FROM (
    SELECT DISTINCT gpi_cd,cd_type
    FROM (&gpi_t1)
) A 
WHERE gpi_cd NOT IN ('', 'UNK', 'NA', 'NaN', 'NAN');

--find out the minimum and maximum index for the gpi table
SELECT * FROM (&gpi_t1_idx) LIMIT 10;

SELECT MIN(idx), MAX(idx)
FROM (&gpi_t1_idx);

select count(*) from (&gpi_t1_idx);

-------------- create final medical index code table 
DROP TABLE IF EXISTS &ihan_idx&iter;

CREATE TABLE &ihan_idx&iter AS
SELECT diag_cd AS cd_value, cd_type, idx FROM (&dx_t1_idx)
UNION
SELECT proc_cd, cd_type, idx FROM (&proc_t1_idx)
UNION
SELECT rvnu_cd, cd_type, idx FROM (&rvnu_t1_idx)
UNION
SELECT gpi_cd,cd_type, idx FROM (&gpi_t1_idx);

SELECT MIN(idx), MAX(idx) FROM (&ihan_idx&iter);

--------- MERGE INDEX TABLE TO GET DESCRIPTION OF THE MEDICAL CODES.This table later will also be used in interpretation and building the IHAN formatting data -------
DROP TABLE IF EXISTS &ihan_final_idx&iter;

CREATE TABLE &ihan_final_idx&iter AS
SELECT DISTINCT
    f.idx,
    f.cd_type,
    f.cd_value,
    i.description
FROM &ihan_idx&iter f
JOIN &input_tbl i    
ON f.cd_type = i.type 
AND f.cd_value = UPPER(TRIM(i.value));

SELECT MIN(idx), MAX(idx) FROM (&ihan_final_idx&iter);

------------ cleanup
drop table IF EXISTS &dx_t1;
drop table IF EXISTS &dx_t1_idx;
drop table IF EXISTS &proc_t1;
drop table IF EXISTS &proc_t1_idx;
drop table IF EXISTS &rvnu_t1;
drop table IF EXISTS &rvnu_t1_idx;
drop table IF EXISTS &gpi_t1;
drop table IF EXISTS &gpi_t1_idx;





