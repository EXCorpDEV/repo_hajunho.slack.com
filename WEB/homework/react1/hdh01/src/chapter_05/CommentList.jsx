import React from "react";
import Comment from "./Comment";

const comments = [
  {
    name: "한덕현",
    comment: "안녕하세요 쏘닥입니다."
  },
  {
    name: "하준호",
    comment: "사무실에 씽크대 들여주세요!"
  },
  {
    name: "사지은",
    comment: "대박나면 고양이 키우게해주세요~"
  }
];

function CommentList(props) {
  return (
    <div>
      {comments.map((comment) => {
        return (
          <Comment name={comment.name} comment={comment.comment} />
        );
      })}
    </div>
  );
}

export default CommentList;