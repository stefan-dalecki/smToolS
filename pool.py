from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing.sharedctypes import copy
import threaded_processing


def processing(script, exporty, writer, num_processes, filelist, task):
    assert num_processes > 1
    assert script.filetype == 'csv'
    # threaded_processing.main(filelist[0], True, script, exporty)
    p = Process(target=task, args=(filelist[0], True, script, exporty, writer))
    p.start()
    p.join()
    filelist = filelist[1:]
    while filelist:
        if num_processes >= len(filelist):
            num_processes = len(filelist)
        with Pool(num_processes) as pool:
            subfiles = filelist[:num_processes]
            bools = [script.display for i in range(num_processes)]
            scripts = [script for i in range(num_processes)]
            exportys = [exporty for i in range(num_processes)]
            writers = [writer for i in range(num_processes)]
            pool.starmap(task, zip(subfiles, bools, scripts, exportys, writer))
            pool.close()
            poo.join()
    filelist = filelist[num_processes:]
