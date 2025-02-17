
import os
import argparse

# coding=utf-8
import os
import time
import datetime
import json
WORK_DIR = os.getcwd()
TOP_DIR = WORK_DIR

def GenCsv(csvName:str, rows:list):
    f = open(csvName, 'w')
    for row in rows:
        rowstr = ''
        for field in row:
            if rowstr:
                rowstr += ','
            rowstr += str(field)
        f.write(rowstr+'\n')
    f.close()
def parse_args():
    parser = argparse.ArgumentParser(description='mmsummary')
    
    parser.add_argument(
        '--ops',
        choices=['summary', 'summarydir', 'summarypaper', 'close_train', 'expdata','expposedata', 'summarypose', 'summaryposedir', 'summaryposepaper'],
        #default='summarydir',
        help='operation')
    parser.add_argument(
        '--file',
        default='./',
        help='file path')
    parser.add_argument(
        '--extargs',
        default='',
        help='extargs')
    args = parser.parse_args()
    return args
def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"文件 {filename} 删除成功！")
    else:
        print(f"文件 {filename} 不存在。")
def ReplaceFileStr(filename, replaceStr:dict):
    f = open(filename, 'r', encoding='utf8')
    dataOld = f.read()
    data = dataOld
    f.close()
    for k, v in replaceStr.items():
        data = data.replace(k, v)
    bk_filename = 'bk_'+filename
    delete_file(bk_filename)
    os.rename(filename, bk_filename)
    f = open(filename, 'w', encoding='utf8')
    f.write(data)
    f.close()
    return dataOld
def Summary(modelFile, img_size = 256, showStruct=False, vitType = None):  
    from mmengine.analysis import get_model_complexity_info
    from mmpretrain import get_model
    repaceDict = {}
    repaceDict["IMG_SIZE = "] = "IMG_SIZE = %s#"%(str(img_size))
    if vitType:
        repaceDict["VIT_SUB_TYPE = "] = "VIT_SUB_TYPE = '%s'#"%(vitType)
    dataOld = None
    if repaceDict and os.path.isfile('mmbase_cfg.py'):
        dataOld = ReplaceFileStr('mmbase_cfg.py', repaceDict)
    input_shape = (3, img_size, img_size)
    model = get_model(modelFile, head=None, neck=None,).backbone
    analysis_results = get_model_complexity_info(model, input_shape)
    if not dataOld:
        print('model', model)

    if showStruct:
        print(analysis_results['out_table'])
        print(analysis_results['out_arch'])
    fname = modelFile.replace('\\', '/').split('/')[-1]
    vitname = fname.replace('config_', '').split('.')[0]
    if repaceDict:
        for k, v in repaceDict.items():
            vitname += '_'+v.split(' ')[-1].split('#')[0].replace("'", "")
    dumpstr = ''
    dumpstr += '%15s,%15s\n'%(vitname+' FLOPs',vitname+' Param')
    dumpstr += '%15s,%15s'%(analysis_results['flops_str'], analysis_results['params_str'])
    print(dumpstr)
    dumpdict = {'flops_str':analysis_results['flops_str'], 'params_str':analysis_results['params_str']}
    json_str = json.dumps(dumpdict)
    f = open('summary.json', 'w', encoding='utf8')
    f.write(json_str)
    f.close()

    if dataOld:
        delete_file('mmbase_cfg.py')
        os.rename('bk_mmbase_cfg.py', 'mmbase_cfg.py')
        #f = open('mmbase_cfg.py', 'w', encoding='utf8')
        #f.write(dataOld)
        #f.close()
    return analysis_results
    
def SummaryDir(dirName = './', files = None):
    if not files:
        files = os.listdir(dirName)
    #print('files', files)
    ret = {}
    vittype_list = ['small', 'base']
    imgsize_list = [256, 384, 512]
    for fname in files:
        realname = dirName+'/'+fname
        if fname.find('config_') != 0:
            continue
        if fname.find('old') >= 0:
            continue
        for vittype in vittype_list:
            if vittype not in ret:
                ret[vittype] = {}
            for imgsize in imgsize_list:
                if imgsize not in ret[vittype]:
                    ret[vittype][imgsize] = {}
                #imgsize = 256
                #vittype = 'base' 
                args = '%d,%s'%(imgsize, vittype)
                os.system('python mmsummary.py --ops summary --file %s --extargs %s'%(fname, args))
                f = open('summary.json', 'r', encoding='utf8')
                json_str = f.read()
                f.close()
                analysis_results = json.loads(json_str)
                #ret.append((fname+'_'+args, analysis_results))
                ret[vittype][imgsize][fname] = analysis_results
    header = ['imagesize', 'vittype',]
    for vittype in vittype_list:
        for imgsize in imgsize_list:
            for (fname, analysis_results) in ret[vittype][imgsize].items():
                vitname = fname.replace('config_', '').split('.')[0]
                header.append('{} FLOPs'.format(vitname))
                header.append('{} Param'.format(vitname))
            break
        break
    data=[]
    
    print(header)
    data.append(header)
    for vittype in vittype_list:
        for imgsize in imgsize_list:
            row = [imgsize, vittype]
            for (fname, analysis_results) in ret[vittype][imgsize].items():
                vitname = fname.replace('config_', '').split('.')[0]
                row.append('{}'.format(analysis_results['flops_str'].replace('G', '')))
                row.append('{}'.format(analysis_results['params_str'].replace('M', '')))
            data.append(row)
    print(data)
    GenCsv('result_mmsummary.csv', data)
def SummaryPose(modelFile):
    os.system('python tools/mmpose_get_flops.py %s --shape 256 --not-print-per-layer-stat'%(modelFile))
    return True
def SummaryPoseDir(dirName = './', files = None):
    if not files:
        files = os.listdir(dirName)
    #print('files', files)
    ret = {}
    vittype_list = ['small', 'base']
    imgsize_list = [256, 384, 512]
    for fname in files:
        realname = dirName+'/'+fname
        if fname.find('config_') != 0:
            continue
        if fname.find('old') >= 0:
            continue
        SummaryPose(fname)
        f = open('summarypose.json', 'r', encoding='utf8')
        json_str = f.read()
        f.close()
        analysis_results = json.loads(json_str)
        #ret.append((fname+'_'+args, analysis_results))
        ret[fname] = analysis_results
                
    header = ['posetype', 'FLOPs', 'Param']

    data=[header]
    
    print(header)
    for (fname, analysis_results) in ret.items():
        vitname = fname.replace('config_', '').replace('.py', '')
        row = [vitname]
        row.append('{}'.format(analysis_results['flops_str'].replace('G', '')))
        row.append('{}'.format(analysis_results['params_str'].replace('M', '')))
        data.append(row)
    print(data)
    GenCsv('pose_mmsummary.csv', data)
#Summary('./config_ci2pvit.py')
CMD_OUTPUT = '/tmp/cmdoutput.txt'

def RunPsCmd(cmd, dumpOut = True):
    if dumpOut:
        cmd = '%s > %s'%(cmd, CMD_OUTPUT)
    print(cmd)
    os.system(cmd)
    if dumpOut:
        f = open(CMD_OUTPUT, 'r')
        data = []
        for line in f.readlines():
            if line and line[len(line) - 1] == '\n':
                line = line[0:len(line) - 1]
            args = []
            for m in line.split(' '):
                if m:
                    args.append(m)
            #print(len(args), args)
            eachArgs = {
                'username':args[0],
                'PID':args[1],
                'CPU':args[2],
                'MEM':args[3],
                'RSS':args[5],
                'COMMAND':' '.join(args[10:]),
            }
            #print(eachArgs)
            data.append(eachArgs)
        f.close()
        return data
    return []
def RunCmd(cmd, dumpOut = True):
    if dumpOut:
        cmd = '%s > %s'%(cmd, CMD_OUTPUT)
    print(cmd)
    os.system(cmd)
    if dumpOut:
        f = open(CMD_OUTPUT, 'r')
        data = f.read()
        return data
    return ''
def close_train(username = 'zxf'):
    cmd = f'ps aux -w w|grep {username}'
    ret = RunPsCmd(cmd)
    #print(ret)
    for eachLine in ret:
        if eachLine['username'] != username:
            continue
        #print(eachLine)
        checkKill = False
        if eachLine['COMMAND'].find('mmsummary.py') >=0:
            continue
        if eachLine['COMMAND'].find('.sh') >=0 and eachLine['COMMAND'].find('train') >=0:
            checkKill = True
        elif eachLine['COMMAND'].find('.py') >=0 and eachLine['COMMAND'].find('train') >=0:
            checkKill = True
        if checkKill:
            cmd = "kill {}".format(eachLine['PID'])
            print(eachLine)
            RunCmd(cmd, False)
        
    return

def GetUserName():
    cmd = f'pwd'
    ret = RunCmd(cmd)
    args = ret.split('/')
    #print(ret, args)
    usernamme = ''
    if ret.find('/home') >= 0:
        usernamme = args[2]
    else:
        usernamme = args[1]
    return usernamme

def expdata(path):
    path = path.replace('\\', '/')
    cfgdirs = os.listdir(path)
    #print('expdata cfgs', cfgdirs)
    for cfgdir in cfgdirs:
        cfgdirreal = path+'/'+cfgdir
        if not os.path.isdir(cfgdirreal):
            continue
        subdirs = os.listdir(cfgdirreal)
        #print('subdir', subdirs)
        for subdir in subdirs:
            if subdir.find('20') < 0:
                continue
            subdirreal = cfgdirreal + '/' + subdir
            if not os.path.isdir(subdirreal):
                continue
            subfilename = subdirreal + '/vis_data' + '/scalars.json'
            cmd = 'python ../tools/vis_tool.py -json %s'%(subfilename)
            print(cmd)
            os.system(cmd)
            
        
    return
def expposedata(path):
    path = path.replace('\\', '/')
    cfgdirs = os.listdir(path)
    #print('expdata cfgs', cfgdirs)
    for cfgdir in cfgdirs:
        cfgdirreal = path+'/'+cfgdir
        if not os.path.isdir(cfgdirreal):
            continue
        subdirs = os.listdir(cfgdirreal)
        #print('subdir', subdirs)
        for subdir in subdirs:
            if subdir.find('20') < 0:
                continue
            subdirreal = cfgdirreal + '/' + subdir
            if not os.path.isdir(subdirreal):
                continue
            subfilename = subdirreal + '/vis_data' + '/scalars.json'
            cmd = 'python ../tools/vis_tool.py -ops pose -json %s'%(subfilename)
            print(cmd)
            os.system(cmd)
            
        
    return
def main():
    args = parse_args()
    ops = args.ops
    print('ops', ops)
    if ops == 'summary':
        extargs = args.extargs.split(',')
        imgsize = 256
        vittype = 'base'
        if len(extargs) >= 2:
            imgsize = int(extargs[0])
            vittype = extargs[1]
        Summary(args.file, imgsize, False, vittype)
    elif ops == 'summarydir':
        SummaryDir(args.file)
    elif ops == 'summarypaper':
        SummaryDir(args.file, ['config_vitp16.py', 'config_ci2pvit.py', 'config_ci2pvitplus.py'])#
    if ops == 'summarypose':#python  ..\ci2pvit\mmsummary.py --ops summarypose --file config_ci2pvit.py
        SummaryPose(args.file, True)
    elif ops == 'summaryposedir':
        SummaryPoseDir(args.file)
    elif ops == 'summaryposepaper':
        SummaryPoseDir(args.file, ['config_ci2pvit.py', 'config_ci2pvit_focus.py', 'config_mobilenet.py',
                                   'config_focus.py', 'config_myvitvar.py', 'config_myvitvar_plu.pys', 
                                   'config_myvitvar_focus.py', 'config_myvitvar_plus_focus.py'])#
        
    elif ops == 'expdata':
        expdata(args.file)
    elif ops == 'expposedata':
        expposedata(args.file)
    elif ops == 'close_train':
        username = GetUserName()
        print('username', username)
        close_train(username)

if __name__ == '__main__':
    main()
