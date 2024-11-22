function onOpenCvReady() {
    cv['onRuntimeInitialized']=()=>{
        //La etiqueta del video de input
        let video = document.querySelector('#videoInput');
        //Usamos la api de camara: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        //Si se puede acceder a la camara, ponemos ese input como src de la etiqueta video
        .then((stream) => {
            video.srcObject = stream;
            video.play();
        })
        //En caso contrario
        .catch((err) => {
            console.log("No se ha podido iniciar la camara: " + err);
        });

        let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
        let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
        let gray = new cv.Mat();
        let cap = new cv.VideoCapture(video);
        let faces = new cv.RectVector();
        let classifier = new cv.CascadeClassifier();
        const FPS = 30;

        //Hacemos peticion al clasificador preentrenado
        fetch('./haarcascade_frontalface_default.xml')
        .then(response => response.arrayBuffer())
        .then(buffer => new Uint8Array(buffer))
        //Lo guardamos en la memoria
        .then(data => cv.FS_createDataFile('/', './haarcascade_frontalface_default.xml', data, true, false, false))
        //Cargamos ese clasificador preentrenado
        .then(() => {
            classifier.load('./haarcascade_frontalface_default.xml');
        })
        //Procesamos el video
        .then(() => processVideo())
        .catch(err => {
            console.log("Error: " + err);
        });

        function processVideo() {
            try {
                // Obtener la fecha de hoy
                let begin = Date.now();
                // Empezar el proceso de captura.
                cap.read(src);
                src.copyTo(dst);
                cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
                // detectar caras.
                classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
                // Pintar los cuadritos sobre las caras.
                for (let i = 0; i < faces.size(); ++i) {
                    let face = faces.get(i);
                    let point1 = new cv.Point(face.x, face.y);
                    let point2 = new cv.Point(face.x + face.width, face.y + face.height);
                    cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
                }
                //Mostrar la imagen en el canvas
                cv.imshow('canvasOutput', dst);
                // Programar la siguiente.
                let delay = 1000/FPS - (Date.now() - begin);
                setTimeout(processVideo, delay);
            } catch (err) {
                console.log("Error: " + err);
            }
        };
    };
  }