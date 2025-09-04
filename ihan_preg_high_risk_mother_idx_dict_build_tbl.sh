source ~/.bashrc     # password. otherwise use -P in $snowsql 
ACNT=carelon-edaprod1.privatelink
USER=AN587958AD
#USER=AN410717AD

TARGET=./ihan_preg_high_rsik_mother_idx_dict_build_tbl.sql
LOG=ihan_high_risk_mother_run_idx_dict.log

#To build a code-index dictionary using the provided input training table data
# Run the query with specified parameters
#the built idx table used to create ihan format data for both training and OOT data. 
# The final output index tables are specified by parameters prefixed with ihan: ihan_final_idx
# for example, NON_CRTFD_AIFS.DL_TS_STAR.ihan_high_risk_preg_mother_medical_code_idx,  

iter="iter_4_v1_training"

snowsql \
    -a $ACNT \
    -u $USER \
    -w DL_AIFS_STAR_USER_WH_L \
    -s DL_TS_STAR_NOGBD \
    -d NON_CRTFD_AIFS \
    -o output_file=$LOG \
    -o friendly=true \
    -o variable_substitution=true \
    -o timing_in_output_file=true \
    -o echo=true \
    -o exit_on_error=true \
    -D iter=$iter \
    -D input_tbl=NON_CRTFD_AIFS.DL_TS_STAR.piad_tmp_wgs_cob_high_risk_mom_final_df_iter_4_v1_traning \
    -D dx_t1=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_dx_t1  \
    -D dx_t1_idx=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_dx_idx  \
    -D proc_t1=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_proc_t1  \
    -D proc_t1_idx=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_proc_idx  \
    -D rvnu_t1=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_rvnu_t1  \
    -D rvnu_t1_idx=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_rvnu_idx  \
    -D gpi_t1=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_gpi_t1  \
    -D gpi_t1_idx=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_gpi_idx  \
    -D ihan_idx=NON_CRTFD_AIFS.DL_TS_STAR.ihan_high_risk_preg_mother_medical_code_idx_ \
    -D ihan_final_idx=NON_CRTFD_AIFS.DL_TS_STAR.ihan_high_risk_preg_mother_medical_code_idx_final_  \
-f ${TARGET}