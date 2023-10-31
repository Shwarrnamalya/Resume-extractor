pip install requests beautifulsoup4 pandas
import requests
from bs4 import BeautifulSoup
import pandas as pd


url = 'https://www.glassdoor.com/Job/united-states-jobs-SRCH_IL.0,13.htm'


response = requests.get(url)

if response.status_code == 200:
    
    soup = BeautifulSoup(response.text, 'html.parser')

    
    job_listings = []

    
    for job in soup.find_all('div', class_='job-listing'):
        url =job.find('span',class='link')
        position = job.find('span', class_='job-title').text
        company = job.find('div', class_='job-info').text
        location = job.find('span', class_='job-location').text
        description = job.find('div', class_='job-description').text

        job_data = {
            'url':      url
            'Position': position,
            'Company': company,
            'Location': location,
            'Job_Description': description
        }

        job_listings.append(job_data)

    
    df = pd.DataFrame(job_listings)

    
    df.to_csv('job_final.csv', index=False)

else:
    print('Failed to retrieve the web page.')

