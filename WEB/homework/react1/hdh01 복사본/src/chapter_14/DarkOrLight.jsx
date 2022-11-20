import { useState, useCallback } from "react";
import ThemeContext from "./ThemeContext";
import MainContent from "./MainContent";

function DarkOrLight(props){
  const [theme, setTheme] = useState("light");

  const toggleThem = useCallback(() =>{
    if (theme == "light") {
      setTheme("dark");
    } else if (theme == "dark"){
      setTheme("light");
    }
  }, [theme]);

  return(
    <ThemeContext.Provider value={{theme, toggleThem}}>
      <MainContent />
    </ThemeContext.Provider>
  );
}

export default DarkOrLight;