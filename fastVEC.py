import os
import argparse

def write_chimera_command(command_file, map1, map2, r_vector, t_vector, output_name_1, output_name_2):
	command_file = open(command_file, 'w')
	command_file.write('from chimera import runCommand as rc\n\n')
	command_file.write('rc("open #0' + map1 + '")\n')
	command_file.write('rc("open #1' + map2 + '")\n')

	# rotate map
    command_file.write('rc(volume #0 voxelSize 0.3)\n')
    command_file.write('rc(vop gaussian #0 sdev 0.3 model #2)\n')
    command_file.write('rc(volume #2 voxelSize 0.3)\n')
    command_file.write('rc(vop gaussian #1 sdev 0.3 model #3)\n')

	# save transformed map
	# command_file.write('rc("vop #1 resample onGrid #0")\n')
    command_file.write('rc("volume #2 save ' + output_name_1 + '")\n')
	command_file.write('rc("volume #3 save ' + output_name_2 + '")\n')
	command_file.close()


# commands to run
# volume #0.1 voxelSize 0.1428
# vop gaussian #0.1 sdev 0.1 model #2