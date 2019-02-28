import requests
import json
import math
import sys
from time import sleep

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

year = 2016

url = 'https://www.moonboard.com/Problems/GetProblems'

data = "sort=&filter=" + ("setupId~eq~'1'" if year == 2016 else "")

headers = {
'Connection': 'keep-alive',
'Content-Length': '39',
'Accept': '*/*',
'Origin': 'https://www.moonboard.com',
'X-Requested-With': 'XMLHttpRequest',
'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/70.0.3538.77 Chrome/70.0.3538.77 Safari/537.36',
'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
'Referer': 'https://www.moonboard.com/Problems/Index',
'Accept-Encoding': 'gzip, deflate, br',
'Accept-Language': 'en,en-US;q=0.9,it-IT;q=0.8,it;q=0.7,de;q=0.6,en-GB;q=0.5',
}

cookies = { '_MoonBoard': 'WJGnMBiKzg5fDyljW8fsk2EjqF-qfKx2_ehp-Pf3zzzG2z0ojxOa871W153IHli_njrvCMZDPYHaQGVas5j-GTzNAzmxAShP8BRpBEPBbAB4hspyipFhYEfoKUcmX959gipq3JJhPofjkreefiZQxVf6bLC1DERtGBkVKKcrm3IzfKEcQdqfh9rUznQN_Mbzq0onzQnJFo_OCluKyWvJ0ZWsLEAMeMnG5MJ4XwyqzcFl8ca0_fw1fH_fVuMSFrzjRG8WesiDpCXFf6ZEd123CZ-MH6VtVemLbpeAqkm7-cfH4Kbv00yEl-Ni27tHWRSMoz2b9uAInQdo5lHi3LbKW_Ae4ytqpiaoykyB3hhPHqMp4MfPHau2cVi74V6652dQu7_bRMXVKdOwv6JiBwzT0s0--dkHCEGTj7yoqZ-eW0bARRqeXwOX3El9r13jjOIOnvZppSsfsm4GHRjXvqRUsDAx88bczmJJ0UdXFrDklVBqeU62GcPNx2BgaFmEWN_B_P-pHEafoPzmzwemsEnDig', '.AspNet.TwoFactorRememberBrowser': '3HCQw0KQ3VVMoGga7RTr6TR68jmiq5Gvg_LBtwMOOMi74oP5SORcdLhsMJvkYUQ7VJP9UvML5cq-1chuWPaBXhW5CjvuA2gLKRz-t-ZfI0fOaM-OF-TNuCyDYh9es6Hb6ZTH7zE4QnrnToauIa7cxOqO8EHz1OFPiBsF30BleQq3W1PTlc80igMS7GLI6lTmkRDYU-zLWlEe1F4IJI7MkdIhhgi-O-PAGLQSjpDL3FrwM-lnRGFdrnbf-7l7ekG83O4F0Fl__TUubOdxKvSEEHvg5V3GLwhHepQtmmLt5VaBaYjjkWHMu0HEdQFBTAeVuiyPaTX7ppqW6BHKCqh_2e46tHlTf1wkDX9eAU5nhAU', '__RequestVerificationToken': '_m0v5kjFUTqQ-orGqY5v6SF03qs8OEUdRB3ThylUTNbgxNC52pKuuyGinI7gpSMpcZygMRKUBOwZ1wYpEkiDKrsbyGaBBvmQio9k8qZMDJo1' }

req = requests.post(url, data=data, headers=headers, cookies=cookies)
res = json.loads(req.text)

n = res["Total"]
page_size = 15 # default from website, should be safe...
n_iterations = math.ceil(float(n) / page_size)
problem_list = []

print("Moonboard edition:", year)
#print("Number of problems:", n)

printProgressBar(0, n_iterations, prefix = 'Downloading data:', length = 50)

for i in range(n_iterations):
	req = requests.post(url, data=data + "&page=" + str(i + 1) + "&pageSize=" + str(page_size), headers=headers, cookies=cookies)
	try:
		res = json.loads(req.text)
		problem_list += res["Data"]
	except json.decoder.JSONDecodeError:
		pass
	# Update progress bar
	printProgressBar(i + 1, n_iterations, prefix = 'Downloading data:', length = 50)
	# wait half a second, we don't want to DoS the website :P
	sleep(0.5)

print(str(len(problem_list)), "problems written in", sys.argv[1])

with open(sys.argv[1], 'w') as outfile:
    json.dump(problem_list, outfile, indent=1)
