#!/bin/bash

#
#for round in {20..30..10}
#do
#    for nodes_probe in {20..30..10}
#    do
#    	#echo 'terminal rounds:' $round 'Nodes to probe: ' $nodes_probe
#    	output=test_results/stochastic_5000/test_R${round}_Net_10_prob_${nodes_probe}
#    	python3 main.py --terminal_round $round --testing_episode 1000 --nodes_to_probe $nodes_probe --net_part_vis 10 >> $output
#    done
#done
#
python3 tl_main.py --gpu_id 1 --net_part_vis 0.1 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_bm_CEL_EW_04_Prb20/tst_DRL_UN_rnd_st_bm_CEL_mv_0_1_N_0_1_Prb_20_R_10 &
wait
python3 tl_main.py --gpu_id 1 --net_part_vis 0.2 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_bm_CEL_EW_04_Prb20/tst_DRL_UN_rnd_st_bm_CEL_mv_0_1_N_0_2_Prb_20_R_10 &
wait
python3 tl_main.py --gpu_id 1 --net_part_vis 0.3 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_bm_CEL_EW_04_Prb20/tst_DRL_UN_rnd_st_bm_CEL_mv_0_1_N_0_3_Prb_20_R_10 &
wait
python3 tl_main.py --gpu_id 1 --net_part_vis 0.4 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_bm_CEL_EW_04_Prb20/tst_DRL_UN_rnd_st_bm_CEL_mv_0_1_N_0_4_Prb_20_R_10 &
wait
python3 tl_main.py --gpu_id 1 --net_part_vis 0.5 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_bm_CEL_EW_04_Prb20/tst_DRL_UN_rnd_st_bm_CEL_mv_0_1_N_0_5_Prb_20_R_10 &
wait
python3 tl_main.py --gpu_id 1 --net_part_vis 0.7 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_bm_CEL_EW_04_Prb20/tst_DRL_UN_rnd_st_bm_CEL_mv_0_1N_0_7_Prb_20_R_10 &
wait
python3 tl_main.py --gpu_id 1 --net_part_vis 1 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_bm_CEL_EW_04_Prb20/tst_DRL_UN_rnd_st_bm_CEL_mv_0_1_N_1_Prb_20_R_10 &
wait
#python3 tl_main.py --gpu_id 1 --net_part_vis 2 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_bm_CEL_EW_04_Prb20/tst_DRL_UN_rnd_st_bm_CEL_1_N_2_Prb_20_R_10 &
#wait
python3 tl_main.py --gpu_id 1 --net_part_vis 3 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_bm_CEL_EW_04_Prb20/tst_DRL_UN_rnd_st_bm_CEL_mv_0_1_N_3_Prb_20_R_10 
#wait
#python3 main.py --gpu_id 0 --net_part_vis 4 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_tl_CEL_WC/tst_DRL_UN_rnd_st_tl_CEL_1_N_4_Prb_10_R_10 &
#wait
#python3 main.py --gpu_id 0 --net_part_vis 5 --testing_episode 2000 >> ./test_results/ca-GrQc/res_rnd_st/res_tl_CEL_WC/tst_DRL_UN_rnd_st_tl_CEL_1_N_5_Prb_10_R_10 
#wait
#python3 main.py --gpu_id 0 --net_part_vis 6 --testing_episode 2000 >> ./test_results/facebook_combined/res_rnd_st/tst_DRL_UN_rnd_st_mv_0_1_N_6_Prb_10_R_10 & 
#wait
#python3 main.py --gpu_id 0 --net_part_vis 7 --testing_episode 2000 >> ./test_results/facebook_combined/res_rnd_st/tst_DRL_UN_rnd_st_mv_0_1_N_7_Prb_10_R_10 & 
#wait
#python3 main.py --gpu_id 0 --net_part_vis 8 --testing_episode 2000 >> ./test_results/facebook_combined/res_rnd_st/tst_DRL_UN_rnd_st_mv_0_1_N_8_Prb_10_R_10 & 
#wait
#python3 main.py --gpu_id 0 --net_part_vis 9 --testing_episode 2000 >> ./test_results/facebook_combined/res_rnd_st/tst_DRL_UN_rnd_st_mv_0_1_N_9_Prb_10_R_10 &
#wait
#python3 main.py --gpu_id 0 --net_part_vis 10 --testing_episode 2000 >> ./test_results/facebook_combined/res_rnd_st/tst_DRL_UN_rnd_st_mv_0_1_N_10_Prb_10_R_10 
