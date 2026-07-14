onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/clk_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/rst_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/hart_id_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/irq_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/flush_i_valid_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/flush_i_ready_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/inst_addr_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/inst_cacheable_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/inst_data_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/inst_valid_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/inst_ready_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/acc_qreq_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/acc_qvalid_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/acc_qready_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/acc_prsp_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/acc_pvalid_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/acc_pready_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/data_req_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/data_rsp_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/ptw_valid_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/ptw_ready_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/ptw_va_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/ptw_ppn_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/ptw_pte_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/ptw_is_4mega_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/fpu_rnd_mode_o}
add wave -noupdate -divider {Shared DIV}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_hive[0]/i_snitch_hive/i_snitch_shared_muldiv/i_div/id_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_hive[0]/i_snitch_hive/i_snitch_shared_muldiv/i_div/operator_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_hive[0]/i_snitch_hive/i_snitch_shared_muldiv/i_div/op_a_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_hive[0]/i_snitch_hive/i_snitch_shared_muldiv/i_div/op_b_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_hive[0]/i_snitch_hive/i_snitch_shared_muldiv/i_div/in_vld_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_hive[0]/i_snitch_hive/i_snitch_shared_muldiv/i_div/in_rdy_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_hive[0]/i_snitch_hive/i_snitch_shared_muldiv/i_div/out_vld_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_hive[0]/i_snitch_hive/i_snitch_shared_muldiv/i_div/out_rdy_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_hive[0]/i_snitch_hive/i_snitch_shared_muldiv/i_div/id_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_hive[0]/i_snitch_hive/i_snitch_shared_muldiv/i_div/res_o}
add wave -noupdate -divider {fpnew div}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/hart_id_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/operands_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/is_boxed_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/rnd_mode_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/op_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/op_mod_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/src_fmt_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/dst_fmt_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/int_fmt_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/vectorial_op_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/tag_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/simd_mask_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/in_valid_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/in_ready_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/flush_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/result_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/status_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/extension_bit_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/tag_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/out_valid_o}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/out_ready_i}
add wave -noupdate {/tb_bin/fix/i_snitch_cluster/i_cluster/gen_core[0]/i_snitch_cc/gen_fpu/i_snitch_fp_ss/i_fpu/i_fpu/gen_operation_groups[1]/i_opgroup_block/gen_merged_slice/i_multifmt_slice/busy_o}
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {2200163 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 150
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 1
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ns
update
WaveRestoreZoom {0 ps} {2730 ns}
