import requests

# Set your Google Custom Search JSON API key and search engine ID
api_key = 'AIzaSyBkEO5llHX0EHh8kSHHT_ghA91M5NWrCNk'
search_engine_id = '53ccc6eef7ace491c'

# Function to perform a web search and fetch results
def search_web(query):
    url = f'https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print('Error occurred:', response.status_code)
        return None

# Input query
query = input('Enter your search query: ')

# Call the function to fetch search results
results = search_web(query)

# Extract details from search results
if results:
    items = results.get('items', [])
    if len(items) > 0:
        for item in items:
            title = item.get('title')
            link = item.get('link')
            snippet = item.get('snippet')
            publication_date = item.get('pagemap', {}).get('metatags', [{}])[0].get('pubdate')
            image_thumbnail = item.get('pagemap', {}).get('cse_thumbnail', [{}])[0].get('src')
            print('Title:', title)
            print('Link:', link)
            print('Snippet:', snippet)
            print('Publication Date:', publication_date)
            print('Image Thumbnail:', image_thumbnail)
            print('---')
    else:
        print('No results found.')
