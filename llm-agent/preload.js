// const { contextBridge, ipcRenderer } = require("electron");

// contextBridge.exposeInMainWorld("api", {
// 	sendMessage: () => {
// 		const chatBox = document.getElementById("chat-box");
// 		ipcRenderer.send("send-message", chatBox.value);
// 		chatBox.value = "";
// 	},
// 	getFriends: () => {
// 		return ipcRenderer.invoke("get-friends");
// 	}
// })

// window.addEventListener("DOMContentLoaded", () => {
// 	ipcRenderer.send("main-window-ready");
// })

const { contextBridge, ipcRenderer, net } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  activateApp: () => ipcRenderer.send('activate-app')
});

contextBridge.exposeInMainWorld("api", {
	sendFormData: ()=>{
		const node=document.createElement('div');
		const innerText=document.createElement('p');
		const input=document.getElementById('chat-box').value;
		if (input!=""){
		const textnode = document.createTextNode(input);
		innerText.classList.add('user-text');
		node.classList.add('user-text-container')
		innerText.appendChild(textnode);
		node.appendChild(innerText)
		document.getElementById("content").appendChild(node);
		if (document.getElementById('welcome')){
			document.getElementById('welcome').remove();
		}
		document.getElementById('chat-box').value="";
		return input;
	}
	},
	checkFormData: ()=>{
		document.getElementById('chat-box').value="";
		document.getElementById('chat-box').placeholder="Get started"
	},
	getData: (prompt)=> {
		const url = "http://localhost:8000/send_prompt";
		fetch(url,
			{
				method: 'POST',
				headers: {
					'Content-Type':'application/json',
					'accept': 'application/json'
				},
				body:JSON.stringify({
					content:prompt
				})
			}
		)
		.then(response=>{
			 return response.json();
		})
		.then(data=>{
			const machine_text=data.message;
			const noder=document.createElement('div');
			const textnode = document.createTextNode(machine_text);
			noder.classList.add('machine-text');
			noder.appendChild(textnode);
			if (data.url){
				const machine_img_url=data.url;
				const img_elem=document.createElement('img');
				img_elem.src=machine_img_url;
				noder.appendChild(img_elem);
			}
			const logo=document.createElement('img');
			logo.src='./Type 1.gif'
			logo.classList.add('side-logo');
			const test=document.createElement('div');
			test.classList.add('machine-container')
			test.appendChild(logo)
			test.appendChild(noder)
			test.style.display='flex';
			document.getElementById("content").appendChild(test)
		})
	  }
})
window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
window.SpeechGrammarList = window.SpeechGrammarList || window.webkitSpeechGrammarList;
window.SpeechRecognitionEvent = window.SpeechRecognitionEvent || window.webkitSpeechRecognitionEvent;




	