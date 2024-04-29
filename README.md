1. Run ```python driver_st.py``` for spatiotemporal env. relevent files are: env_3d.py, gp_st_ipp.py, /fire_commander/catnipp_2d_fire.py, and attention_net.py

2. Result from training is stored in gifs/ipp/

3. Trying out mamba based encoder in place of self-attention encoder in attention_mamba. Result (with single layer) not significantly different (from the loss curves. tests are remaining). Make sure to check the import attention_net in driver_st, runner, and worker_st files to switch between vanilla attention_net and attention_mamba. 

4. Copied STAMP (target tracking work from Sartoretti's lab) architecture to attention_net_st. Created {driver, runner, worker_}@stamp.py for using this. This accounts for the history of states, has temporal encoder (cross-attention) as well. Setting target_size (num_agents) to 1 should make it work i think. Currently getting some ray::IDLE issue; all CPU cores get full even before a single episode ends (irrespective of setting num_cpus=1 within ray.init()). The issue is likely to not be related to Ray- i can comment out the Ray parts of the code (driver_stamp --> runner_stamp --> worker_stamp) and still see the memory leak/ whatever issue it is (this is basically with just one job).

5. GP hyperparam tuning is v important. I think we need to increase both spatial and temporal hp. (this is in gp_st_ipp.py). Maybe not spatial (not sure), but definitely temporal hp.

6. Need to think about possibility of using Recurrent Neural Process.

7. Can Attentive Kernel be brought back into the picture? Training might be tricky though.

8. Need to read more on robustness and policy transfer between different environments.