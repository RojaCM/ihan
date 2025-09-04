source ~/.bashrc     # password. otherwise use -P in $snowsql 
ACNT=carelon-edaprod1.privatelink
USER=AN587958AD


TARGET=./ihan_preg_high_risk_mother_data_givenIndexTbl.sql
LOG=ihan_preg_high_risk_mother_OOT_data_givenIndexTbl.log

#To build a code-index dictionary using the provided input training table data
# Run the query with specified parameters iter and input tbl
#the built idx table used to create ihan format data for both training and OOT data. 
# The final output index tables are specified by parameters prefixed with ihan: ihan_final_idx
# for example, NON_CRTFD_AIFS.DL_TS_STAR.ihan_high_risk_preg_mother_medical_code_idx_final_iter4_v3_training,  

iter="iter_4_v1_OOT"

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
    -D max_num_code_per_visit=512  \
    -D iter=$iter \
    -D input_tbl=NON_CRTFD_AIFS.DL_TS_STAR.piad_tmp_wgs_cob_high_risk_mom_final_df_iter_4_v1_OOT \
    -D dx_t1=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_dx_t1  \
    -D dx_t1_idx=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_dx_idx  \
    -D proc_t1=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_proc_t1  \
    -D proc_t1_idx=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_proc_idx  \
    -D rvnu_t1=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_rvnu_t1  \
    -D rvnu_t1_idx=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_rvnu_idx  \
    -D gpi_t1=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_gpi_t1  \
    -D gpi_t1_idx=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_gpi_idx_final_  \
    -D ihan_final_idx=NON_CRTFD_AIFS.DL_TS_STAR.ihan_high_risk_preg_mother_medical_code_idx_final_iter_4_v1_training \
    -D dx_t3=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_dx_t3  \
    -D proc_t3=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_proc_t3  \
    -D rvnu_t3=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_rvnu_t3  \
    -D gpi_t3=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_gpi_t3  \
    -D final_dx_data=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_dx_data  \
    -D final_proc_data=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_proc_data  \
    -D final_rvnu_data=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_rvnu_data  \
    -D final_gpi_data=NON_CRTFD_AIFS.DL_TS_STAR.ag14432_high_risk_preg_mother_gpi_data  \
    -D static_tbl=NON_CRTFD_AIFS.DL_TS_STAR.high_risk_mom_static_feat_table \
    -D ihan_tbl=NON_CRTFD_AIFS.DL_TS_STAR.ihan_high_risk_preg_mother_final_data_ \
-f ${TARGET}