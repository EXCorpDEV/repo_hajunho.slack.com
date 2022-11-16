import React, {useState} from "react";
import Toolbar from "./Toolbar";

function LandingPage(props) {
  const [isLoggeIn, setIsLoggedIn] = useState(false);

  const onClickLogin = () => {
    setIsLoggedIn(true);
  };

  const onClickLogout = () => {
    setIsLoggedIn(false);
  };

  return (
    <div>
      <Toolbar 
        isLoggedIn={isLoggeIn}
        onClickLogin={onClickLogin}
        onClickLogout={onClickLogout}
      />
      <div style={{padding: 16}}>Sodoc simple on-line Doctor</div>
    </div>
  );
}

export default LandingPage;