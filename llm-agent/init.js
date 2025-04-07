document.getElementById("chat-form").addEventListener("submit", (e) => {
	e.preventDefault();
	window.api.sendMessage();
})
// document.getElementById("get-friends").addEventListener("click", async (e) => {
// 	const friends = await window.api.getFriends();
// 	const friendList = document.getElementById("friend-list");
// 	friendList.innerHTML = "";
// 	friendList.append(...friends.map(x => {
// 		const li = document.createElement("li");
// 		li.innerText = `[${x.id}] ${x.name}`;
// 		return li;
// 	}));
// })
document.getElementById("nav-btn").addEventListener("click", async (e) => {
	let w=document.getElementById("nav").style.display;
	console.log(w);
	if (w!="none" || w==""){
		document.getElementById("nav").style.display="none";
	}else{
		document.getElementById("nav").style.display="flex";
	}
})

let history={};

document.getElementById("send-chat").addEventListener("click", async (e) => {
	e.preventDefault();
	const prompt=window.api.sendFormData();
	console.log(prompt);
	if (prompt!=undefined){
	ResponseText(prompt)
	}
})	

function ResponseText(prompt){
	window.api.getData(prompt)
}
document.addEventListener('DOMContentLoaded', function () {
	const chatBox = document.getElementById('chat-box');
	chatBox.addEventListener('input', function () {
		autoExpand(chatBox);
	});
});

function autoExpand(textarea) {
	textarea.style.height = 'auto';
	textarea.style.height = textarea.scrollHeight + 'px';
}

document.getElementById("chat-box").addEventListener("focus", async (e) => {
	document.getElementById("dispose").style.flex='0';
	document.getElementById('chat-form').style.maxWidth="100%";
	document.getElementById('chat-box').placeholder="Type your prompt here..."
})

document.getElementById("chat-box").addEventListener("focusout", async (e) => {
	setTimeout(()=>{
		console.log(document.getElementById('content').innerHTML);
		if (document.getElementById('content').innerHTML==""){
			document.getElementById("dispose").style.flex='1';
			document.getElementById('chat-form').style.maxWidth="40vw";
			window.api.checkFormData();
	}},200)
})
// document.addEventListener('DOMContentLoaded', () => {
//     const form = document.getElementById('chat-form');
//     const inputField = document.getElementById('chat-box');

//     inputField.addEventListener('keydown', (event) => {
//         if (event.key === 'Enter') {
//             event.preventDefault();  // Prevent the default action (form submission) on Enter key press
//             form.submit();           // Manually submit the form
//         }
//     });
// });
// document.getElementById('chat-form').addEventListener('submit', async(e)=>{
// 	setTimeout(()=>{
// 		e.preventDefault();
// 		window.api.sendFormData();
// 	})
// })