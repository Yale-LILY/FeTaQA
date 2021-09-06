import json
import sys
import glob
import os

if __name__=="__main__":
    if len(sys.argv)<3:
        print('usage: python dataset_format.py inputdir outputdir')
        exit(1)
    globpath = os.path.join(sys.argv[1],'*.jsonl')
    for infile in glob.glob(globpath):
        with open(infile) as f:
            lines = [json.loads(l) for l in f]
        basename = infile.split('/')[-1].split('.')[0]
        split = basename.split('_')[-1]
        version = basename.split('_')[0].split('-')[-1]
        out = {'version':version, 'split':split, 'data': lines}
        outfile = os.path.join(sys.argv[2],f'{basename}.json')
        with open(outfile,'w') as f:
            json.dump(out,f)


        
    
