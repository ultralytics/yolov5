function startStream() {
   
    fetch('/start_stream')
        .then(response => response.json())
        .then(data => {
            console.log('Start Stream:', data);
            document.getElementById('videoStream').src = '/video_feed';  
        });
}

function stopStream() {

    fetch('/stop_stream')
        .then(response => response.json())
        .then(data => {
            console.log('Stop Stream:', data);
            document.getElementById('videoStream').src = ''; 
        });
}

  
document.addEventListener('DOMContentLoaded', function() {
    const videoDropdown = document.getElementById('videoDropdown');
    //const playButton = document.getElementById('playButton');
    // const detectionButton = document.getElementById('detectionButton');
    const videoPlayer = document.getElementById('videoPlayer');
    detectionButton= document.querySelector(".detectionButton");
    playButton= document.querySelector(".playButton");
    
    

    fetch('/videos').then(response => response.json()).then(videos => {
        videos.forEach(video => {
            let option = document.createElement('option');
            option.value = video;
            option.textContent = video;
            videoDropdown.appendChild(option);
        });
    });

    videoDropdown.onchange = function() {
        if (this.value) {
            playButton.style.display = 'block';
            detectionButton.style.display='block'
            
        } else {
            playButton.style.display = 'none';
            detectionButton.style.display = 'none';
        }
    };

    playButton.onclick = function() {

        videoPlayer.innerHTML = `<video id = vi src="/video/${videoDropdown.value}" controls></video>`;
        // function playVideo(videoSource) {
        //     var videoPlayer = document.getElementById("videoPlayer");
        //     videoPlayer.src = videoSource;
        //     videoPlayer.load();
        //     videoPlayer.play();
        //     toggleDropdown(); // Close the dropdown after selecting a video
        //   }
        //   playVideo(`/videos/${videoDropdown.value}`);
        
    };

    detectionButton.onclick = function() {
        
        this.innerHTML="<div class= 'loader'></div>" 
        // setTimeout(()=>{
        //     this.innerHTML="Detection Done";
        //     this.style="background : #f1f5f4; color: #333; pointer-events: none";
        // },2000)
        
        fetch(`/detection/${videoDropdown.value}`).then(response => response.json()).then(data => {
            this.innerHTML="Detection Done";
            this.style="background : #f1f5f4; color: #333; pointer-events: none";
            alert(data.status);
        })
        .catch(error => {
            
            loadingMessage.style.display = 'none';
            console.error('Error:', error);
        });
    };
});
