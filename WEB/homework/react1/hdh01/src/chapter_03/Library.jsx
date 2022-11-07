import React from "react";
import Book from "./Book";

function Library(props) {
  return(
    <div>
      <Book name="처음 만난 파이썬" numOPage={300} />
      <Book name="처음 만난 AWS" numOPage={400} />
      <Book name="처음 만난 리액트" numOPage={500} />
    </div>
  );
}

export default Library;
