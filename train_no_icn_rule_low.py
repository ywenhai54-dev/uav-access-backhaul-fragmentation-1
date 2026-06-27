from train_core_rule_low_fpsfix_v21 import run_main

# run_main applies force_training_safe_diagnostics, including sc_pfr_probe_mode=off.
if __name__ == '__main__':
    run_main('no_icn', '训练 A2: w/o ICN（规则低层版 FPSFix v21）')
