# Movies Recommendation System

A web application that displays user profiles and movie listings, with the ability to track user watches for future machine learning recommendations using TensorFlow.js.

## Project Structure

- `index.html` - Main HTML file for the application
- `index.js` - Entry point for the application
- `view/` - Contains classes for managing the DOM and templates
- `controller/` - Contains controllers to connect views and services
- `service/` - Contains business logic for data handling
- `data/` - Contains JSON files with user and movie data

## Setup and Run

1. Install dependencies:
```
npm install
```

2. Start the application:
```
npm start
```

3. Open your browser and navigate to `http://localhost:8080`

## Features

- User profile selection with details display
- Past purchase history display
- Movie listing with "Watch Now" functionality
- Purchase tracking using sessionStorage

## Future Enhancements

- TensorFlow.js-based recommendation engine
- User similarity analysis
- Movie recommendation based on watch list
