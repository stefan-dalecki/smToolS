from multiprocessing import Pool
from multiprocessing import Process


def processing(script, num_processes, filelist, task):
    assert num_processes > 1
    assert script.filetype == 'csv'
    p = Process(target=task, args=(filelist[0], True, script))
    p.start()
    p.join()
    filelist = filelist[1:]
    while filelist:
        with Pool(num_processes) as pool:
            subfiles = filelist[:num_processes]
            bools = [script.display for i in range(num_processes)]
            scripts = [script for i in range(num_processes)]
            pool.starmap(task, zip(subfiles, bools, scripts))
            pool.close()
    filelist = filelist[num_processes:]
