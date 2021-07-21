from chimera import runCommand as rc
rc("open #0' + map1 + '")
rc("open #1' + map2 + '")

	# rotate map
rc("volume #0 voxelSize 0.3")
rc("vop gaussian #0 sdev 0.3 model #2")
rc("volume #2 voxelSize 0.3")
rc("vop gaussian #1 sdev 0.3 model #3")
rc("vop #1 resample onGrid #0")
rc("volume #2 save ' + output_name_1 + '")
rc("volume #3 save ' + output_name_2 + '")