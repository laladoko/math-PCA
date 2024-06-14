// let BobInList = userList.some(a => a.name == "Bbb")
// console.log(BobInList)
// let like = 0;
// for(let i=0; i<posts.length;i++){
//     like = like + posts[i].likes;
    
// }
// console.log(like)
// let reduce = posts.reduce((sum,post)=>post.likes+sum,0)
// console.log(reduce)
// let like = 0;
// posts.forEach(
//     post =>{
//         like += post.likes;
//     }
// )
// console.log(like)
// posts.sort((a,b)=>a.date-b.date)
// console.log(posts)
let users = {
    user1: { name: "Alice", age: 25, followers: 120 },
    user2: { name: "Bob", age: 30, followers: 150 },
    user3: { name: "Charlie", age: 35, followers: 80 }
  };


  const UserList = ({ users }) => { // 接受一个包含users的props对象
    return (
      <ul>
        {Object.values(users).map((user, index) => ( // 正确调用map函数
          <li key={index}>
            <p>{user.name}</p>