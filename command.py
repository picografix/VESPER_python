import os
import argparse


def input(cmd):
    parser = argparse.ArgumentParser(description='Run Vesper')
    parser.add_argument('-i', '--filename', required=True, action='store', dest='filename',
                        help='Required. Name of the reference map file.')
    parser.add_argument('-a', '--file1', required=True, action='store', dest='file1',
                        help='Required. Name of the target map file.')
    # parser.add_argument('-b', required = True, action = 'store', dest = 'vesper_result', help = 'Required. Name of the result file from VESPER.')
    # parser.add_argument('-odir', action = 'store', dest = 'out_dir', help = 'Optional. Directory for the transformed target map files. If not specified, the transformed target map files would be written to the current directory')
    parser.add_argument('-b', '--file2', action='store', dest='file2', help='Required. Name of the  file 2')
    parser.add_argument('-c', '--Nthr', action='store', dest='Nthr', help='[int  ] : Number of cores for threads')
    parser.add_argument('-t', '--th1', action='store', dest='th1', help=' [float] : Threshold of density map1')
    parser.add_argument('-T', '--th2', action='store', dest='th2', help='[float] : Threshold of density map2')
    parser.add_argument('-g', '--dreso', action='store', dest='dreso',
                        help='[float] : Bandwidth of the Gaussian filter')
    parser.add_argument('-A', '--ang', action='store', dest='ang', help='[float] : Sampling angle spacing def=30.0')
    parser.add_argument('-N', '--TopN', action='store', dest='TopN', help=' [int  ] : Refine Top [int] models def=10')
    parser.add_argument('-R', '--LocalR', action='store', dest='LocalR',
                        help='Required. Name of the reference map file.')
    parser.add_argument('-k', '--Dkeep', action='store', dest='Dkeep', help='Required. Name of the reference map file.')
    parser.add_argument('-r', '--Nround', action='store', dest='Nround',
                        help='Required. Name of the reference map file.')
    parser.add_argument('-l', '--Ntabu', action='store', dest='Ntabu', help='Required. Name of the reference map file.')
    parser.add_argument('-s', '--ssize', action='store', dest='ssize', help='[float] : Sampling voxel spacing def=7.0')
    parser.add_argument('-S', '--ShowGrid', action='store', dest='ShowGrid',
                        help='Required. Name of the reference map file.')
    parser.add_argument('-V', '--Mode1', action='store', dest='Mode1', help='Vector Products Mode')
    parser.add_argument('-L', '--Mode2', action='store', dest='Mode2', help='Overlap Mode')
    parser.add_argument('-C', '--Mode3', action='store', dest='Mode3', help='Cross Correlation Coefficient Mode')
    parser.add_argument('-P', '--Mode4', action='store', dest='Mode4',
                        help='Pearson Correlation Coefficient Mode def=false')
    parser.add_argument('-F', '--Mode5', action='store', dest='Mode5', help='Laplacian Filtering Mode def=false')
    parser.add_argument('-E', '--Emode', action='store', dest='Emode',
                        help='Evaluation mode of the current position def=false')

    # arg parser
    args = parser.parse_args()

    # define some values
    cmd.filename = args.filename
    cmd.file1 = args.file1
    cmd.file2 = args.file2

    # if values provided
    if args.Nthr:
        cmd.Nthr = int(args.Nthr)
    if args.dreso:
        cmd.dreso = (args.dreso)
    if args.LocalR:
        cmd.LocalR = args.LocalR
    if args.Dkeep:
        cmd.Dkeep = args.Dkeep
    if args.Nround:
        cmd.Nround = args.Nround
    if args.Ntabu:
        cmd.Ntabu = args.Ntabu
    if args.ssize:
        cmd.ssize = args.ssize
    if args.ang:
        cmd.ang = args.ang
    if args.TopN:
        cmd.TopN = args.TopN
    if args.ShowGrid:
        cmd.ShowGrid = args.ShowGrid

    if args.th1:
        cmd.th1 = args.th1
    if args.th2:
        cmd.th2 = args.th2

    if args.Mode1:
        cmd.Mode = 1
    elif args.Mode2:
        cmd.Mode = 2
    elif args.Mode3:
        cmd.Mode = 3
    elif args.Mode4:
        cmd.Mode = 4
    elif args.Mode5:
        cmd.Mode = 5
    if args.Emode:
        cmd.Emode = True
