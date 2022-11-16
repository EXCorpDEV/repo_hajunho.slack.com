import React from "react";

const students = [
  {
    id: 1,
    name: "hyun"
  },
  {
    id: 2,
    name: "jun"
  },
  {
    id: 3,
    name: "sa"
  },
  {
    id: 4,
    name: "inje"
  }
]
function AttendanceBook(props) {
  return (
    <ul>
      {students.map((student) => {
        return<li key={student.id}>{student.name}</li>; //id를 키값으로 사용
      })}
    </ul>
  );
}

export default AttendanceBook;