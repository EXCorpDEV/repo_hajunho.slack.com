import React from 'react';
import { ReactDOM } from 'react';
import { useEffect } from 'react';

const UseEffectApp = () => {
  const sayHello = () => console.log("hello");

  const [number, setNumber] = useState(0);
  const [aNumber, setAnumber] = useState(0);

  useEffect(sayHello, [number]);

  return (
    <>
    <div className="UseEffectApp">
      <div>Hi</div>
      <button onClick={() => setNumber(number + 1)}>{number}</button>
      <button onClick={() => setAnumber(aNumber + 1)}>{aNumber}</button>
    </div>
    </>
  );
};

const App = () => (
  <div>
    <UseEffectApp>Hello, Webpack!</UseEffectApp>
  </div>
);

export default App;
