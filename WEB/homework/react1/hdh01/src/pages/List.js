import React from 'react';

const User = ({userData}) => {
  return (
    <tr>
      <td>{userData.name}</td>
      <td>{userData.email}</td>
    </tr>
  )
}

const UserList = () => {
  const users = [
    {email: 'ha@gmail.com', name: '하준호'},
    {email: 'han@gmail.com', name: '한덕현'},
    {email: 'sa@gmail.com', name: '사지은'},
    {email: 'chun@gmail.com', name: '춘식이'}
  ];

  return (
    <table>
      <thead>
        <tr>
          <th>이름</th>
          <th>이메일</th>
        </tr>
      </thead>
      <tbody>
        {users.map(user => <User userData={user} />)}
      </tbody>
    </table>
  )
}

export default UserList;