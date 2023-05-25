from threading import Thread
import time
from main import get_args,main


if __name__ == '__main__':
    args = get_args()
    num_clients = args.num_clients
    threads = []
    for i in range(num_clients):
        threads.append(Thread(target=main,args=(args,i)))
    
    for thread in threads:
        thread.start()
        time.sleep(20)


