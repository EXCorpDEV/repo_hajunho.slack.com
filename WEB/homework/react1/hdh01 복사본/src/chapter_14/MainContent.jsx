import {useContext} from "react";
import ThemeContext from "./ThemeContext";

function MainContent(props){
  const {theme, toggleThem} = useContext(ThemeContext);

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        padding: "1.5rem",
        background: theme == "light" ? "white" : "black",
        color: theme == "light" ? "black" : "white",
      }}
    >
      <p>안녕하세요, 테마 변경이 가능한 웹사이트 입니다.</p>
      <button onClick={toggleThem}>테마 변경</button>
    </div>
  );
}

export default MainContent;