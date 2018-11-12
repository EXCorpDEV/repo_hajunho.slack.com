//
//  jhscene.swift
//  bridge8
//
//  Created by Junho HA on 2020. 2. 22..
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

open class jhScene : jhSceneScrollView, UIGestureRecognizerDelegate, observer_p {
    
    public func jhRedraw() {
    }

    var jhEnforcingMode: Bool = true
    
    var scenePanPoint = CGPoint()
    
    var mPanels : [jhPanel<jhScene>]
    
    /// for logging
    var countTouch : UInt = 0
    
    struct iPhoneScreensize {
        var width : CGFloat
        var height : CGFloat
        init(_ width : CGFloat, _ height : CGFloat) {
            self.width = width
            self.height = height
        }
    }
    
    required public init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override public init(frame: CGRect) {
        
        self.mPanels = Array<jhPanel>()
        
        super.init(frame: frame)
        
        super.jhSceneFrameWidth = frame.width
        super.jhSceneFrameHeight = frame.height
        
        self.guideLine.zPosition = 10
        self.isDirectionalLockEnabled = true
//        self.guideLine.addSublayer(guideLine)
        //        layer.frame = CGRect(x: 0, y: 0, width: 100 , height: 100 )
        
        
        
//        let swipeRight = UISwipeGestureRecognizer(target: self, action: #selector(respondToSwipeGesture))
//        swipeRight.direction = UISwipeGestureRecognizerDirection.right
//        let swipeLeft = UISwipeGestureRecognizer(target: self, action: #selector(respondToSwipeGesture))
//        swipeLeft.direction = UISwipeGestureRecognizerDirection.left
//        let swipeUp = UISwipeGestureRecognizer(target: self, action: #selector(respondToSwipeGesture))
//        swipeUp.direction = UISwipeGestureRecognizerDirection.up
//        let swipeDown = UISwipeGestureRecognizer(target: self, action: #selector(respondToSwipeGesture))
//        swipeDown.direction = UISwipeGestureRecognizerDirection.down
//
//        // ViewController will be the delegate for the left and right swipes
//        swipeRight.delegate = self
//        swipeLeft.delegate = self
//        swipeDown.delegate = self
//        swipeUp.delegate = self
//
//        self.addGestureRecognizer(swipeRight)
//        self.addGestureRecognizer(swipeLeft)
//        self.addGestureRecognizer(swipeUp)
//        self.addGestureRecognizer(swipeDown)
        
        "".pwd(self)
        jhDataCenter2.attachObserver(observer: self)
    }
    
  

 
//    @objc func respondToSwipeGesture(gesture: UIGestureRecognizer) {
//        if let swipeGesture = gesture as? UISwipeGestureRecognizer {
//            switch swipeGesture.direction {
//            case UISwipeGestureRecognizerDirection.right:
//                print("Swiped right")
//            case UISwipeGestureRecognizerDirection.down:
//                print("Swiped down")
//            case UISwipeGestureRecognizerDirection.left:
//                print("Swiped left")
//            case UISwipeGestureRecognizerDirection.up:
//                print("Swiped up")
//            default:
//                break
//            }
//        }
//    }
    
    func jhColor(r:CGFloat , g:CGFloat , b:CGFloat , a:Float) -> CGColor {
        return  UIColor(red: r / 255.0, green: g / 255.0, blue: b / 255.0, alpha: r).cgColor
    }
    
    open func drawScene() {
        drawPanels()
    }
    
    open func drawPanels() {
//        super.addSubview(view)
        for x in mPanels {
            super.addSubview(x)
        }
    }
    
    public func createPanels(s : jhScene, withHeightRatios: ratioNtype...) {
        
        var panel : jhPanel<jhScene>? = nil
        var y : CGFloat = 0.0
        var vHeight : CGFloat = 0.0
        
        "".pwd(self)
        
        for rnt in withHeightRatios {
            if(jhGS.s.logLevel.contains(.graphPanel)) { print("createPanels(withHeightRatios: CGFloat...)", rnt)}
            
            assert(!(rnt.ratio < 0.1 || rnt.ratio > 10.0), "heightRation Range is 0.1~10.0")
            
            vHeight = rnt.ratio * 0.1 * self.jhSceneFrameHeight
            panel = jhGraphBuilder<jhScene>()
                .type(rnt.type)
                .frame(0, y, jhSceneFrameWidth*4, vHeight)
                .scene(self)
                .build()
            y += vHeight
            
            if jhGS.s.logLevel.contains(.graphPanel) {
                print("jhScene_addPanel_mHeightStack =", vHeight, "\n y = \(y) heightRatio = \(rnt)")
            }
            panel!.backgroundColor = UIColor.white
            mPanels.append(panel!)
            panel = nil
            }
    }
    
    override open func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        print("touchesMove(_:)", countTouch)
        
        countTouch += 1
        if jhGS.s.logLevel.contains(.graph2) { print("jhScene_touchesMoved", countTouch) }
        
        let touch = (touches as NSSet).anyObject()!
        let current = (touch as AnyObject).location(in: self)
        
        
    }
    
    override open func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        countTouch += 1
        if jhGS.s.logLevel.contains(.graph2) { print("jhScene_touchesBegan", countTouch) }
        
        if jhGS.s.logLevel.contains(.graph2) { print("jhScene_touchesMoved", countTouch) }
        
        let touch = (touches as NSSet).anyObject()!
        let current = (touch as AnyObject).location(in: self)
        
        guideLine.removeFromSuperlayer()
//        guideLine = jhGuideLine(x: current.x, y: current.y, lineWidth: 1, layer: 0)!
        guideLine = jhGuideLine(x: current.x, y: 0, lineWidth: 1, layer: 0)!
        guideLine.frame = CGRect(x: 0, y: 0, width: self.jhSceneFrameWidth*4, height: self.contentSize.height) //TODO: will be changed.
        guideLine.zPosition=1
        //        guideLine.isGeometryFlipped = true
        guideLine.backgroundColor = UIColor(white: 1, alpha:0.1).cgColor
        self.layer.addSublayer(guideLine)
        guideLine.setNeedsDisplay()
    }
    
    override open func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        countTouch += 1
        if jhGS.s.logLevel.contains(.graph2) { print("jhScene_touchesEnded", countTouch) }
    }
    
//    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
//        if (gestureRecognizer is UIPanGestureRecognizer || gestureRecognizer is UIRotationGestureRecognizer) {
//            return true
//        } else {
//            return false
//        }
//    }
    
    // here are those protocol methods with Swift syntax
    func gestureRecognizer(gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWithGestureRecognizer otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        return true
    }
    func gestureRecognizer(gestureRecognizer: UIGestureRecognizer, shouldReceiveTouch touch: UITouch) -> Bool {
        return true
    }
  
}
