function inicio(){
    alert("Tengo que hacer el login");
}

function login(){
    var c = "gtg1";
    var u = "gtg";
    if(document.form.password.value == c && document.form.user.value == u){
        alert("Usario y contraseña correctas");
        window.location= "video.html";
    } else{
        alert("Usuario y/o contraseña incorrectos");
    }
}
