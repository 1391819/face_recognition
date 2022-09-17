<div align="center">

<img src="/Utilities/logo.png" alt="logo" width="128"/>

</div>

<h1 align="center">Face Recognition System</h1>

<div align="justify">

## About

A face recognition system implemented for my [Diploma di Stato](https://qips.ucas.com/qip/italy-diploma-di-esame-di-stato-conclusivo-dei-corsi-di-istruzione-secondaria-superiore) project. After starting the prototype through the attached button, the camera continuously scans the environment for people's faces by using the Viola-Jones algorithm. Following this, the Eigenface method (PCA based) is used for face recognition, which can be trained and then used to successfully identify people.

## Stack

- [Raspberry Pi 3](https://www.raspberrypi.org/) - A small single-board computer
- [Raspbian (Jessie)](https://www.raspbian.org/) - A Debian based OS optimised for the Raspberry Pi hardware
- [SQLite3](https://www.sqlite.org/index.html) - A C library that provides a lightweight disk-based database that doesn't require a separate server process
- [OpenCV](https://opencv.org/) - An open source computer vision and machine learning software library
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

## Roadmap

- [x] Single face - detection and recogntion
- [x] Multiple faces - detection and recognition
- [ ] Improve external design
- [ ] Translate thesis to English
- [ ] Clear folders structure

## Highlights

  <div align="center">
    <img src="/Utilities/screenshots/schedule.jpg" alt="gantt chart"/>
    <br/>
    <br/>
    <img src="/Utilities/screenshots/pca.jpg" alt="pca"/>
    <br/>
    <br/>
    <img src="/Utilities/screenshots/raspberry_pi.jpg" alt="prototype"/>
    <br/>
    <br/>
    <img src="/Utilities/screenshots/analysis.jpg" alt="analysis"/>
  </div>

## License

Apache 2.0

## Attributions

- <a href="https://www.flaticon.com/free-icons/face" title="face icons">Face icons created by Flat Icons - Flaticon</a>

</div>
