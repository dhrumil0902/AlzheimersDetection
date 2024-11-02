data_path = "..\data\EEG_data\"
group_paths = {"AD\", "Healthy\"}
type_paths = {"Eyes_closed\"}
patient_paths = {"Paciente3\"}

channel_paths = {"C3.txt", "C4.txt", "Cz.txt", "F1.txt", "F2.txt", "F3.txt", "F4.txt", "F7.txt", "F8.txt", "Fp1.txt", "Fp2.txt", "Fz.txt", "O1.txt", "O2.txt", "P3.txt", "P4.txt", "Pz.txt", "T3.txt", "T4.txt", "T5.txt", "T6.txt"}

channels = []

for gp_itr = 1:length(group_paths)
    group_path = group_paths{gp_itr}

    for tp_itr = 1:length(type_paths)
        type_path = type_paths{tp_itr}

        for pp_itr = 1:length(patient_paths)
            patient_path = patient_paths{pp_itr}

            for cp_itr = 1:length(channel_paths)
                channel_path = channel_paths{cp_itr}
                channel_path_full = data_path + group_path + type_path + patient_path + channel_path
                
                channel = load(channel_path_full)
                channels = [channels; transpose(channel)]
            end
        end
    end
    break
end