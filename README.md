<div align="center">

<img src="/Utilities/logo.png" alt="logo" width="128"/>

</div>

<h1 align="center">Face Recognition System</h1>

<div align="justify">

A face recognition system which I implemented for my [Diploma di Stato](https://qips.ucas.com/qip/italy-diploma-di-esame-di-stato-conclusivo-dei-corsi-di-istruzione-secondaria-superiore) project. After starting the prototype through the attached button, the camera continuously scans the environment for people's faces by using the Viola-Jones algorithm. Following this, the Eigenface method (PCA based) is used for face recognition, which can be trained and then used to successfully identify people.

## Roadmap

- [x]  Install and configure the Raspberry Pi (Jessie)
- [x]  Set up an SQLite3 database to store face data
- [x]  Develop a prototype that continuously scans the environment for faces
- [x]  Implement functionality to manage the faces data
- [x]  Integrate the Viola-Jones algorithm for face detection
- [x]  Implement the Eigenface method for face recognition
- [x]  Train the face recognition model
- [x]  Test and evaluate with known and unknown faces
- [x]  Improve performance and optimisation
    - [x]  Optimise the algorithms and code to ensure efficient face data pre-processing and real-time performance on the Raspberry Pi
    - [x]  Fine-tune the parameters of the face detection and recognition algorithms to improve accuracy and speed
- [x]  Extend the existing implementation to handle multiple faces simultaneously
- [x]  Improve user usability (e.g., user management through a button)
- [ ]  Improve external design
- [ ]  Translate thesis to English
- [ ]  Code refactoring

## Stack

- Raspberry Pi 3 (Jessie)
- OpenCV
- SQLite3
- Viola-Jones algorithm
- Eigenface method

## Project structure

```
$PROJECT_ROOT
│   # Presentation files (italian only) and other small testing scripts
├── utilities
│   # Custom face data
├── eigenvectors
│   # Dataset
├── training
│   # Key scripts
└── ...
```

## Highlights

  <div align="center">
    <img src="/Utilities/screenshots/schedule.jpg" alt="Gantt chart"/>
    <br/>
    <br/>
    <img src="/Utilities/screenshots/pca.jpg" alt="PCA"/>
    <br/>
    <br/>
    <img src="/Utilities/screenshots/raspberry_pi.jpg" alt="prototype"/>
    <br/>
    <br/>
    <img src="/Utilities/screenshots/analysis.jpg" alt="analysis"/>
  </div>

## Attributions

- <a href="https://www.flaticon.com/free-icons/face" title="face icons">Face icons created by Flat Icons - Flaticon</a>

## License

[Apache 2.0](https://github.com/1391819/face_recognition/blob/master/License.txt) © [Roberto Nacu](https://github.com/1391819)

</div>
