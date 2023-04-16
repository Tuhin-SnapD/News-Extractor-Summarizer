// JavaScript code for handling button clicks and fetching text files, images, and searching on Google

// Add event listeners to the buttons
document.getElementById('btn1').addEventListener('click', fetchTextAndImage.bind(null, 'business.txt', 'business.png'));
document.getElementById('btn2').addEventListener('click', fetchTextAndImage.bind(null, 'entertainment.txt', 'entertainment.png'));
document.getElementById('btn3').addEventListener('click', fetchTextAndImage.bind(null, 'health.txt', 'health.png'));
document.getElementById('btn4').addEventListener('click', fetchTextAndImage.bind(null, 'science.txt', 'science.png'));
document.getElementById('searchBtn').addEventListener('click', searchOnGoogle);

// Fetch text file and image and display content
function fetchTextAndImage(textFilename, imageFilename) {
  // Fetch text file
  var textXhr = new XMLHttpRequest();
  textXhr.open('GET', 'dataset/final/' + textFilename, true);
  textXhr.onreadystatechange = function() {
    if (textXhr.readyState === XMLHttpRequest.DONE) {
      if (textXhr.status === 200) {
        document.getElementById('content').textContent = textXhr.responseText;
      } else {
        document.getElementById('content').textContent = 'Error fetching text file: ' + textFilename;
      }
    }
  };
  textXhr.send();

  // Fetch image file
  var imgXhr = new XMLHttpRequest();
  imgXhr.open('GET', 'dataset/graphs/' + imageFilename, true);
  imgXhr.responseType = 'blob';
  imgXhr.onreadystatechange = function() {
    if (imgXhr.readyState === XMLHttpRequest.DONE) {
      if (imgXhr.status === 200) {
        var imageURL = URL.createObjectURL(imgXhr.response);
        document.getElementById('image').src = imageURL;
      } else {
        document.getElementById('image').src = '';
      }
    }
  };
  imgXhr.send();
}

// Search on Google with generated text as query
function searchOnGoogle() {
  var query = document.getElementById('content').textContent;
  if (query) {
    var googleSearchUrl = 'https://www.googleapis.com/customsearch/v1?key=AIzaSyBkEO5llHX0EHh8kSHHT_ghA91M5NWrCNk&cx=53ccc6eef7ace491c&q=' + encodeURIComponent(query);
    // Replace 'YOUR_API_KEY' and 'YOUR_CUSTOM_SEARCH_ENGINE_ID' with your actual API key and custom search engine ID
    window.open(googleSearchUrl);
  }
}

