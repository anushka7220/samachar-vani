async function upload(){

const file=document.getElementById("file").files[0]

if(!file){

alert("Please upload a newspaper image")

return

}

const form=new FormData()

form.append("file",file)

document.getElementById("status").innerText=
"Analyzing newspaper..."

const res=await fetch("/upload",{

method:"POST",

body:form

})

const data=await res.json()

checkStatus(data.job_id)

}


async function checkStatus(id){

const interval=setInterval(async()=>{

const res=await fetch("/job/"+id)

const data=await res.json()

if(data.status==="processing"){

document.getElementById("status").innerText=
"Generating podcast..."

}

if(data.status==="completed"){

clearInterval(interval)

document.getElementById("status").innerText=
"Podcast ready"

document.getElementById("result").innerHTML=
`
<audio controls src="/audio/${id}"></audio>
`

}

},3000)

}