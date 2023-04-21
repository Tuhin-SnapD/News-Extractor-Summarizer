// JavaScript code for handling button clicks and fetching text files, images, and searching on Google

// Add event listeners to the buttons
document
  .getElementById("btn1")
  .addEventListener(
    "click",
    fetchTextAndImage.bind(null, "business.txt", "business.png")
  );
document
  .getElementById("btn2")
  .addEventListener(
    "click",
    fetchTextAndImage.bind(null, "entertainment.txt", "entertainment.png")
  );
document
  .getElementById("btn3")
  .addEventListener(
    "click",
    fetchTextAndImage.bind(null, "health.txt", "health.png")
  );
document
  .getElementById("btn4")
  .addEventListener(
    "click",
    fetchTextAndImage.bind(null, "science.txt", "science.png")
  );
document.getElementById("searchBtn").addEventListener("click", searchOnGoogle);

// Fetch text file and image and display content
function fetchTextAndImage(textFilename, imageFilename) {
  // Fetch text file
  var textXhr = new XMLHttpRequest();
  textXhr.open("GET", "../dataset/final/" + textFilename, true);
  textXhr.onreadystatechange = function () {
    if (textXhr.readyState === XMLHttpRequest.DONE) {
      if (textXhr.status === 200) {
        document.getElementById("content").textContent = textXhr.responseText;
      } else {
        document.getElementById("content").textContent =
          "Error fetching text file: " + textFilename;
      }
    }
  };
  textXhr.send();

  // Fetch image file
  var imgXhr = new XMLHttpRequest();
  imgXhr.open("GET", "../dataset/graphs/" + imageFilename, true);
  imgXhr.responseType = "blob";
  imgXhr.onreadystatechange = function () {
    if (imgXhr.readyState === XMLHttpRequest.DONE) {
      if (imgXhr.status === 200) {
        var imageURL = URL.createObjectURL(imgXhr.response);
        document.getElementById("image").src = imageURL;
      } else {
        document.getElementById("image").src = "";
      }
    }
  };
  imgXhr.send();
}

function searchOnGoogle() {
  var query = document.getElementById("content").textContent;
  if (query) {
    var encodedQuery = encodeURIComponent(query); // Encode query text
    var googleSearchUrl =
      "https://www.googleapis.com/customsearch/v1?key=" +
      GOOGLE_API +
      "&cx=" +
      SEARCH_ENGINE_ID +
      "&q=" +
      encodedQuery;
    // Replace 'YOUR_API_KEY' and 'YOUR_CUSTOM_SEARCH_ENGINE_ID' with your actual API key and custom search engine ID

    // Make an API request to Google Custom Search
    // Make an API request to Google Custom Search
    fetch(googleSearchUrl)
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        // Clear previous search results
        document.getElementById("searchResults").innerHTML = "";

        // Check if items property is present in the response
        if (data && data.items && data.items.length > 0) {
          // Loop through search results and display them
          var items = data.items;
          for (var i = 0; i < items.length; i++) {
            var resultItem = document.createElement("div");
            var resultTitle = document.createElement("h3");
            resultTitle.textContent = items[i].title;
            var resultLink = document.createElement("a");
            resultLink.href = items[i].link;
            resultLink.textContent = items[i].link;
            var resultImage = document.createElement("img"); // Create image element
            resultImage.src = items[i].pagemap.cse_thumbnail[0].src;
            // resultImage.src = items[i].pagemap.metatags[0]["og:image"]; // Set image source to og:image property in metatags array
            resultImage.alt = items[i].title; // Set image alt text to title
            resultItem.appendChild(resultImage); // Append image to result item
            resultItem.appendChild(resultTitle);
            resultItem.appendChild(resultLink);
            document.getElementById("searchResults").appendChild(resultItem);
          }
        } else {
          // Display message if no results found
          var noResultsMsg = document.createElement("p");
          noResultsMsg.textContent = "No results found.";
          document.getElementById("searchResults").appendChild(noResultsMsg);
        }
      })
      .catch(function (error) {
        console.error("Error fetching search results:", error);
      });
  }
}
