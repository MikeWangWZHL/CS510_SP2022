touché22-image-search
=====================
Version: 2022-01-21

This dataset contains the collection for Touché 2022 Task 3: Image Retrieval for Arguments. It contains data for 23841 images.

Web: https://webis.de/events/touche-22/

For having a quick look at the data structure, download and extract the touche22-image-search-preview.zip, which contains all data for a single image. The same data is also contained in the other ZIP archives.

The main part of the dataset is distributed as a collection of ZIP archives:
  touche22-image-search-main.zip                  (containing files marked with '*' below)    4 GB  (12 GB uncompressed)
  touche22-image-search-archives.zip              (containing files marked with 'A' below)   81 GB  (81 GB uncompressed)    
  touche22-image-search-nodes.zip                 (containing files marked with 'N' below)    5 GB (191 GB uncompressed)
  touche22-image-search-png-images.zip            (containing files marked with 'P' below)   17 GB  (17 GB uncompressed)
  touche22-image-search-screenshots.zip           (containing files marked with 'S' below)   62 GB  (67 GB uncompressed)

On UNIX, use the command line 'unzip' program to unpack them all into one common directory structure.

The dataset is structured as follows:
```
README.txt                                       This file
images/
  I<first-two-chars-of-image-url-hash>/
    I<full-sixteen-chars-image-url-hash>/        Directory name == image ID
      image.png                              P   Image in PNG format
      image.webp                             *   Image in WebP format
      image-url.txt                          *   URL of the image
      image-phash.txt                        *   64bit pHash of the (WebP) image as string: https://www.phash.org/
      pages/
        P<full-sixteen-chars-page-url-hash>/     Directory name == page ID
          page-url.txt                       *   URL of the web page (containing the image)
          rankings.jsonl                     *   Each line contains a JSON object describing a query to Google that retrieved the image/page as follows:
                                                 {
                                                   "query":"<query text>",
                                                   "topic":"<Touché topic ID for which the query was intended>",
                                                   "rank":<image/page rank in result list starting with 1>
                                                 }
          snapshot/
            dom.html                         *   Snapshot of the HTML DOM
            image-xpath.txt                  *   Each line contains the XPath of a node in the dom.html that references the image (img, picture, or meta in this order)
            nodes.jsonl                      N   Each line contains a JSON object describing a node of dom.html as follows:
                                                 {
                                                   "xPath": "<XPath of the node in the dom.html>",
                                                   "visible": <Boolean whether the node is visible as per https://stackoverflow.com/a/33456469>,
                                                   "classes": ["<entry of the class attribute>", ...],
                                                   "position": [
                                                     <left border of node pixel position in screenshot.webp starting left with 0>,
                                                     <top border of node pixel position in screenshot.webp starting top with 0>,
                                                     <right border of node pixel position in screenshot.webp starting left with 0>,
                                                     <bottom border of node pixel position in screenshot.webp starting top with 0>
                                                   ],
                                                   "text": "text content of the node",
                                                   "css": {
                                                     "<css-attribute>": "<css-attribute value>",
                                                     ...
                                                   }
                                                 }
            screenshot.png                   S   Screenshot of the page in PNG format
            text.txt                         *   Text content of the dom.html (taken from the first node of the nodes.jsonl)
            web-archive.warc.gz              A   Web archive file containing all resources requested when taking the snapshot
```
topics.xml                                       Title, description, and narrative of Touché topics 1 to 50 (see "Topics" below)
training-qrels.txt                               Relevance ratings for a few topics and images (see "Training data" below)


Topics
------
Standard Touché topics 1 to 50 as XML file:
<topics>
  <topic>
    <number>:        The Touché topic ID
    <title>:         The user query for the topic
    <description>:   A short background story for the topic
    <narrative>:     Hints for human annotators on what constitutes a relevant result argument (to be adapted for images)

These topics are used to evaluate approaches.

If you use the description and/or narrative in your approach, specify so in your overview paper.


Training data
-------------
The file training-qrels.txt provides 334 relevance ratings from the ArgMining 2021 paper "Image Retrieval for Arguments Using Stance-Aware Query Expansion" (https://webis.de/publications.html#kiesel_2021e) for images in this dataset. Each line describes a relevance rating as follows (rating 1 = "relevant" and 0 = "not relevant"):
<Touché topic ID> <stance> <image ID> <rating>


Version history
---------------

2022-01-21
  - Removed 1211 images for which different HTML parsers provided different image XPaths. Thanks to team Aramis for detecting these!
  - Removed 5 images from the trainings-qrels accordingly

2021-12-01
  - Removed 13 images where the XPaths were irregular due to faulty HTML (thanks to team Aramis for pointing out this problem!)
  - Removed images from the training data that are no longer in the collection
 
