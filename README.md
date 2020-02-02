# Artathon2020-MyProject
## Inspiration
The main inspiration behind building this art piece is the recent climatic problems that our planet is facing. Humans are one of the main causes behind the bushfires burning across Australia. CNN reported that "NSW police have charged at least 24 people with deliberately starting bushfires, and have taken legal action against 183 people for fire-related offenses since November, according to a police statement." (Jessie Yeung, 2020). The images express the emotions of nature.

## What it does
The artwork will change based on the emotional expression of the person standing in front of the art piece.

## How we built it
The art piece consists of four main collages. The collages present five elements affected by climate change water, land, trees, animals, and air.
The emotion detection is done by analyzing the output produced by the facial landmarks detection model made by dlib (ex. http://dlib.net/face_landmark_detection.py.html).
The final project is made with Python and the OpenCV library.

## Challenges we ran into
We had very limited time to produce the different images needed for conveying the emotions. The pre-trained models we found had some emotions that weren't useful for our case and making a model then training it would take a lot of time, especially because of our limited computing power.

## Accomplishments that we're proud of
The emotion detection algorithm and the artwork. We're also proud of the overall progress made in this small amount of time.
