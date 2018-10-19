//
//  jhScene.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhScene : UIScrollView, jhScene_p, observer_p {
    
    var jhEnforcingMode: Bool = true
    var jhSceneFrameWidth: CGFloat = jhDraw.maxR
    var jhSceneFrameHeight: CGFloat = jhDraw.maxR
    let draw = jhDraw()
    var guideLine : jhGuideLine
    var tempCount : UInt64 = 0
    var mPanels : [jhPanel]
    
    struct iPhoneScreensize {
        var width : CGFloat
        var height : CGFloat
        init(_ width : CGFloat, _ height : CGFloat) {
            self.width = width
            self.height = height
        }
    }
    
    override init(frame: CGRect) {
        self.jhSceneFrameWidth = frame.width
        self.jhSceneFrameHeight = frame.height
        self.guideLine = jhGuideLine(x: 10, y: 10, lineWidth: 1, layer: 0)!
        self.mPanels = Array<jhPanel>()
        
        super.init(frame: frame)
        self.guideLine.zPosition = 10
        //        self.guideLine.addSublayer(guideLine)
        //        layer.frame = CGRect(x: 0, y: 0, width: 100 , height: 100 )
        
        "".pwd(self)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func jhColor(r:CGFloat , g:CGFloat , b:CGFloat , a:Float) -> CGColor {
        return  UIColor(red: r / 255.0, green: g / 255.0, blue: b / 255.0, alpha: r).cgColor
    }
    
    func drawScene() {
        drawPanels()
    }
    
    func drawPanels() {
        //        super.addSubview(view)
        for x in mPanels {
            super.addSubview(x)
        }
    }
    
    func createPanels(withHeightRatios: ratioNtype...) {
        
        var panel : jhPanel? = nil
        var y : CGFloat = 0.0
        var vHeight : CGFloat = 0.0
        
        "".pwd(self)
        
        for rnt in withHeightRatios {
            if(GS.shared.logLevel.contains(.graphPanel)) { print("createPanels(withHeightRatios: CGFloat...)", rnt)}
            
            assert(!(rnt.ratio < 0.1 || rnt.ratio > 10.0), "heightRation Range is 0.1~10.0")
            
            vHeight = rnt.ratio * 0.1 * self.jhSceneFrameHeight
            panel = jhGraphBuilder().type(rnt.type).frame(0, y, jhSceneFrameWidth*4, vHeight).build()
            y += vHeight
            
            if GS.shared.logLevel.contains(.graphPanel) {
                print("jhScene_addPanel_mHeightStack =", vHeight, "\n y = \(y) heightRatio = \(rnt)")
            }
            panel!.backgroundColor = UIColor.white
            mPanels.append(panel!)
            panel = nil
        }
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        tempCount += 1
        if GS.shared.logLevel.contains(.graph2) { print("jhScene_touchesMoved", tempCount) }
        
        let touch = (touches as NSSet).anyObject()!
        let current = (touch as AnyObject).location(in: self)
        
        guideLine.removeFromSuperlayer()
        guideLine = jhGuideLine(x: current.x, y: current.y, lineWidth: 1, layer: 0)!
        guideLine.frame = CGRect(x: 0, y: 0, width: self.jhSceneFrameWidth*4, height: self.jhSceneFrameHeight) //TODO: will be changed.
        guideLine.zPosition=1
        //        guideLine.isGeometryFlipped = true
        guideLine.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(guideLine)
        guideLine.setNeedsDisplay()
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        tempCount += 1
        if GS.shared.logLevel.contains(.graph2) { print("jhScene_touchesBegan", tempCount) }
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        tempCount += 1
        if GS.shared.logLevel.contains(.graph2) { print("jhScene_touchesEnded", tempCount) }
    }
    
    func jhRedraw() {
        
    }
}
